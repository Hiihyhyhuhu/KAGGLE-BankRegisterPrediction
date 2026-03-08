import numpy as np
import pandas as pd
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import cupy as cp

# Scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             precision_score, recall_score, log_loss, 
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# Boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# GPU (RAPIDS)
import cuml
import cudf

# GPU/RAPIDS Support (Handles environment without GPU gracefully)
try:
    import cudf
    import cupy as cp
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("RAPIDS/GPU libraries not found. Falling back to CPU mode.")

class StackingModel(BaseEstimator, ClassifierMixin):
    """
    Advanced Stacking Model with GPU support, Optuna optimization, and comprehensive evaluation.
    Features:
    - GPU acceleration for all base estimators
    - Optuna hyperparameter optimization
    - Meta-feature generation
    - Comprehensive plotting and evaluation
    - Model persistence
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42,
                 device: str = 'cuda', n_jobs: int = -1,
                 meta_learner: str = 'logistic_regression'):
        """
        Initialize StackingModel
        Args:
            n_splits: Number of folds for cross-validation
            random_state: Random seed for reproducibility
            device: Device to use ('cuda' or 'cpu')
            n_jobs: Number of parallel jobs
            meta_learner: Type of meta learner ('logistic_regression', 'xgb)
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.device = 'cuda' if (device == 'cuda' and HAS_GPU) else 'cpu'
        self.n_jobs = n_jobs
        self.meta_learner = meta_learner

        self.base_estimators = {}
        self.trained_base_estimators = {}
        self.meta_features_train = None
        self.meta_learner_model = None
        self.label_encoder = None
        self.is_fitted = False
        self.history = {
            'optuna_trials': {},
            'training_metrics': {},
            'validation_metrics': {}
        }

    def add_base_estimators(self, estimators: Dict[str, Any]) -> None:
        """
        Add base estimators to the stacking model
        Args:
            estimators: Dictionary with estimator names as keys and
                       (estimator_instance, param_space) as values
        """
        for name, (estimator, param_space) in estimators.items():
            self.base_estimators[name] = {
                'estimator': estimator,
                'param_space': param_space,
                'best_params': None
            }

    def _get_gpu_params(self, estimator_name: str) -> Dict[str, Any]:
        """Get GPU-specific parameters for each estimator"""
        gpu_params = {}

        if self.device == 'cuda':
            if 'xgboost' in estimator_name.lower():
                gpu_params.update({'device': 'cuda', 'predictor': 'gpu_predictor'})
            elif 'lightgbm' in estimator_name.lower():
                gpu_params.update({'device': 'gpu'})
            elif 'catboost' in estimator_name.lower():
                gpu_params.update({'task_type': 'GPU', 'devices': '0'})
            elif 'randomforest' in estimator_name.lower():
                ## RAPIDS RF not require  device param
                pass
            elif 'logisticregression' in estimator_name.lower() and self.meta_learner == 'logistic_regression':
                gpu_params.update({})

        return gpu_params

    def optuna_search(self, X: pd.DataFrame, y: pd.Series,
                     estimator_name: str, n_trials: int = 3,
                     cv: int = 5, scoring: str = 'roc_auc') -> Any:
        """
        Perform Optuna hyperparameter optimization
        Args:
            X: Training features
            y: Training target
            estimator_name: Name of estimator to optimize
            n_trials: Number of Optuna trials
            cv: Cross-validation folds
            scoring: Scoring metric
        Returns:
            Best estimator with optimized parameters
        """
        if estimator_name not in self.base_estimators:
            raise ValueError(f"Estimator '{estimator_name}' not found")

        estimator_info = self.base_estimators[estimator_name]
        estimator = estimator_info['estimator']
        param_space = estimator_info['param_space']

        # Ensure binary classification
        if len(np.unique(y)) > 2:
            raise ValueError("Only binary classification is supported")

        # Perform a single train/validation split outside the objective for Optuna evaluation
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X, y, test_size=1/cv, random_state=self.random_state, stratify=y
        )

        # Determine if the estimator is a cuML estimator
        is_cuml_estimator = False
        if self.device == 'cuda':
            if 'randomforest' in estimator_name.lower(): # Check for cuML RandomForest
                is_cuml_estimator = True
        # RAPIDS -> cudf DataFrames/Series
        if self.device == 'cuda' and is_cuml_estimator:
            X_train_opt_gpu = cudf.DataFrame.from_pandas(X_train_opt)
            X_val_opt_gpu = cudf.DataFrame.from_pandas(X_val_opt)
            y_train_opt_gpu = cudf.Series(y_train_opt.values)
            y_val_opt_gpu = cudf.Series(y_val_opt.values)
        # non-RAPIDS -> pandas DataFrames
        else:
            X_train_opt_gpu = X_train_opt
            X_val_opt_gpu = X_val_opt
            y_train_opt_gpu = y_train_opt
            y_val_opt_gpu = y_val_opt

        def objective(trial):
            params = {}

            # Build parameters from param_space
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )

            # Add GPU parameters if applicable
            params.update(self._get_gpu_params(estimator_name))
            params['random_state'] = self.random_state

            # Create estimator with trial parameters, conditionally using cuML or scikit-learn
            if self.device == 'cuda' and 'randomforest' in estimator_name.lower():
                 model = cuRF(**params)
            elif 'xgboost' in estimator_name.lower():
                model = XGBClassifier(**params)
            elif 'lightgbm' in estimator_name.lower():
                model = LGBMClassifier(**params, verbosity=-1)
            elif 'catboost' in estimator_name.lower():
                model = CatBoostClassifier(**params, verbose=0)
            elif 'adaboost' in estimator_name.lower():
                model = AdaBoostClassifier(**params)
            elif 'randomforest' in estimator_name.lower():
                model = RandomForestClassifier(**params)
            else:
                 raise ValueError(f"Unknown estimator type for {estimator_name}")

            # Check if a cuml estimator
            is_cuml_estimator = isinstance(model, (cuml.ensemble.RandomForestClassifier,))

            # Convert data to numpy for non-cuml estimators
            if not is_cuml_estimator and self.device == 'cuda':
                X_train_fit = X_train_opt_gpu.to_numpy() if isinstance(X_train_opt_gpu, cudf.DataFrame) else X_train_opt_gpu
                y_train_fit = y_train_opt_gpu.to_numpy() if isinstance(y_train_opt_gpu, cudf.Series) else y_train_opt_gpu
                X_val_predict = X_val_opt_gpu.to_numpy() if isinstance(X_val_opt_gpu, cudf.DataFrame) else X_val_opt_gpu
            else:
                X_train_fit = X_train_opt_gpu
                y_train_fit = y_train_opt_gpu
                X_val_predict = X_val_opt_gpu

            # Fit and evaluate on the pre-split data
            model.fit(X_train_fit, y_train_fit)

            # Predict probabilities and convert to numpy immediately if needed
            val_proba_all = model.predict_proba(X_val_predict)
            if isinstance(val_proba_all, (cudf.Series, cudf.DataFrame, cp.ndarray)): # Added cp.ndarray for cuML output
                val_proba_all = val_proba_all.to_numpy() if hasattr(val_proba_all, 'to_numpy') else cp.asnumpy(val_proba_all) # Convert cp.ndarray to numpy

            if scoring == 'roc_auc':
                val_proba = val_proba_all[:, 1]
                score = roc_auc_score(y_val_opt.to_numpy(), val_proba)
            elif scoring == 'accuracy':
                val_pred = model.predict(X_val_predict)
                if isinstance(val_pred, (cudf.Series, cudf.DataFrame, cp.ndarray)): # Added cp.ndarray
                    val_pred = val_pred.to_numpy() if hasattr(val_pred, 'to_numpy') else cp.asnumpy(val_pred) # Convert cp.ndarray to numpy
                score = accuracy_score(y_val_opt.to_numpy(), val_pred)
            elif scoring == 'f1':
                val_pred = model.predict(X_val_predict)
                if isinstance(val_pred, (cudf.Series, cudf.DataFrame, cp.ndarray)): # Added cp.ndarray
                    val_pred = val_pred.to_numpy() if hasattr(val_pred, 'to_numpy') else cp.asnumpy(val_pred) # Convert cp.ndarray to numpy
                score = f1_score(y_val_opt.to_numpy(), val_pred)
            elif scoring == 'log_loss':
                score = log_loss(y_val_opt.to_numpy(), val_proba_all)
            return score

        # Run Optuna optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Store results
        self.history['optuna_trials'][estimator_name] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'trials': len(study.trials)
        }

        # Load best parameters to models
        best_params = study.best_params.copy()
        best_params.update(self._get_gpu_params(estimator_name))
        best_params['random_state'] = self.random_state

        if self.device == 'cuda' and 'randomforest' in estimator_name.lower():
            best_estimator = cuRF(**best_params)
        elif 'xgboost' in estimator_name.lower():
            best_estimator = XGBClassifier(**best_params)
        elif 'lightgbm' in estimator_name.lower():
            best_estimator = LGBMClassifier(**best_params, verbosity=-1)
        elif 'catboost' in estimator_name.lower():
            best_estimator = CatBoostClassifier(**best_params, verbose=0)
        elif 'adaboost' in estimator_name.lower():
            best_estimator = AdaBoostClassifier(**best_params)
        elif 'randomforest' in estimator_name.lower():
             best_estimator = RandomForestClassifier(**best_params)

        self.base_estimators[estimator_name]['best_params'] = best_params
        self.base_estimators[estimator_name]['estimator'] = best_estimator

        return best_estimator

    def _generate_meta_feature(self, X: pd.DataFrame, y: pd.Series = None,
                              is_training: bool = True) -> np.ndarray:
        """
        Generate meta-features using out-of-fold predictions
        Args:
            X: Features
            y: Target (required for training)
            is_training: Whether this is for training
        Returns:
            Meta-features array (numpy array)
        """
        n_samples = X.shape[0]
        n_estimators = len(self.base_estimators)
        meta_features = np.zeros((n_samples, n_estimators))

        if is_training:
            if y is None:
                 raise ValueError("y is required for training meta-features")

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                random_state=self.random_state)

            # Convert full training dataset to cudf if device='cuda'
            use_cuml_in_base = any(isinstance(info['estimator'], (cuml.ensemble.RandomForestClassifier,))
                                  for info in self.base_estimators.values())
            if self.device == 'cuda' and use_cuml_in_base:
                X_gpu_full = cudf.DataFrame.from_pandas(X)
                y_gpu_full = cudf.Series(y.values)
            else:
                X_gpu_full = X
                y_gpu_full = y

            for i, (name, info) in enumerate(self.base_estimators.items()):
                estimator = info['estimator']
                oof_preds = np.zeros(n_samples)

                # Check if a cuml estimator
                is_cuml_estimator = isinstance(estimator, (cuml.ensemble.RandomForestClassifier,))

                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)): # Split using original pandas DataFrames
                    # Select data for the fold from the appropriate DataFrame type
                    if self.device == 'cuda' and use_cuml_in_base:
                        X_train_fold = X_gpu_full.iloc[train_idx]
                        X_val_fold = X_gpu_full.iloc[val_idx]
                        y_train_fold = y_gpu_full.iloc[train_idx]
                    else:
                        X_train_fold, X_val_fold = X_gpu_full.iloc[train_idx], X_gpu_full.iloc[val_idx]
                        y_train_fold, y_val_fold = y_gpu_full.iloc[train_idx], y_gpu_full.iloc[val_idx]

                    # Convert data to numpy for non-cuml estimators
                    if not is_cuml_estimator and self.device == 'cuda':
                        X_train_fit = X_train_fold.to_numpy() if isinstance(X_train_fold, cudf.DataFrame) else X_train_fold
                        y_train_fit = y_train_fold.to_numpy() if isinstance(y_train_fold, cudf.Series) else y_train_fold
                        X_val_predict = X_val_fold.to_numpy() if isinstance(X_val_fold, cudf.DataFrame) else X_val_fold
                    else:
                        X_train_fit = X_train_fold
                        y_train_fit = y_train_fold
                        X_val_predict = X_val_fold

                    # Fit the estimator
                    estimator.fit(X_train_fit, y_train_fit)

                    # Predict probabilities and convert back to numpy if needed for meta-features
                    val_preds_all = estimator.predict_proba(X_val_predict)
                    if isinstance(val_preds_all, (cudf.Series, cudf.DataFrame, cp.ndarray)): # Added cp.ndarray for cuML output
                        val_preds_all = val_preds_all.to_numpy() if hasattr(val_preds_all, 'to_numpy') else cp.asnumpy(val_preds_all) # Convert cp.ndarray to numpy

                    # Select probability of the positive class
                    val_preds = val_preds_all[:, 1]
                    meta_features[val_idx, i] = val_preds

                # Store the last trained estimator on the full training data (for prediction)
                if self.device == 'cuda' and not is_cuml_estimator:
                    ## Convert to numpy for non-cuml estimators
                    X_full_np = X_gpu_full.to_numpy() if isinstance(X_gpu_full, cudf.DataFrame) else X_gpu_full
                    y_full_np = y_gpu_full.to_numpy() if isinstance(y_gpu_full, cudf.Series) else y_gpu_full
                    estimator.fit(X_full_np, y_full_np)
                else:
                    estimator.fit(X_gpu_full, y_gpu_full)

                self.trained_base_estimators[name] = estimator

        else:
            X_pred_data = X

            # Convert input data to cudf if device is cuda and any base estimator is cuML
            use_cuml_in_base = any(isinstance(info['estimator'], (cuml.ensemble.RandomForestClassifier,))
                                  for info in self.base_estimators.values())
            if self.device == 'cuda' and use_cuml_in_base:
                X_pred_data_gpu = cudf.DataFrame.from_pandas(X_pred_data)
            else:
                X_pred_data_gpu = X_pred_data

            for i, (name, info) in enumerate(self.base_estimators.items()):
                estimator = self.trained_base_estimators.get(name)
                if estimator is None:
                    raise ValueError(f"Base estimator '{name}' not trained. Call fit() first.")

                # Determine if the current estimator is a cuml estimator
                is_cuml_estimator = isinstance(estimator, (cuml.ensemble.RandomForestClassifier,))

                # Convert to numpy for non-cuml estimators before predicting if on GPU
                if not is_cuml_estimator and self.device == 'cuda':
                    X_predict_np = X_pred_data_gpu.to_numpy() if isinstance(X_pred_data_gpu, cudf.DataFrame) else X_pred_data_gpu
                else:
                    X_predict_np = X_pred_data_gpu

                # Predict probabilities and convert back to numpy if needed for meta-features
                test_preds_all = estimator.predict_proba(X_predict_np)
                if isinstance(test_preds_all, (cudf.Series, cudf.DataFrame, cp.ndarray)): # Added cp.ndarray
                    test_preds_all = test_preds_all.to_numpy() if hasattr(test_preds_all, 'to_numpy') else cp.asnumpy(test_preds_all) # Convert cp.ndarray to numpy
                test_preds = test_preds_all[:, 1]
                meta_features[:, i] = test_preds

        # Meta-features are returned as numpy arrays
        return meta_features

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'StackingModel':
        """
        Fit the stacking model
        Args:
            X: Training features
            y: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        Returns:
            self
        """
        # Handle label encoding for non-numeric targets
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y

        # Generate meta-features
        print("Generating meta-features...")
        self.meta_features_train = self._generate_meta_feature(X, y_encoded, is_training=True)

        # Train meta-learner
        print("Training meta-learner...")
        if self.meta_learner == 'logistic_regression':
            if self.device == 'cuda':
                self.meta_learner_model = cuml.linear_model.LogisticRegression(
                    max_iter=1000
                )
                # Convert meta_features_train to cudf for cuML Logistic Regression
                meta_features_train_fit = cudf.DataFrame(self.meta_features_train)
                y_encoded_fit = cudf.Series(y_encoded)
            else:
                self.meta_learner_model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                )
                # Use numpy arrays for scikit-learn Logistic Regression
                meta_features_train_fit = self.meta_features_train
                y_encoded_fit = y_encoded

        elif self.meta_learner == 'xgb':
            self.meta_learner_model = XGBClassifier(
                random_state=self.random_state,
                **self._get_gpu_params('xgboost')
            )
            meta_features_train_fit = self.meta_features_train
            y_encoded_fit = y_encoded
        elif self.meta_learner == 'lgb':
            self.meta_learner_model = LGBMClassifier(
                random_state=self.random_state, verbosity=-1,
                **self._get_gpu_params('lightgbm')
            )
            meta_features_train_fit = self.meta_features_train
            y_encoded_fit = y_encoded

        # Fit the meta-learner
        self.meta_learner_model.fit(meta_features_train_fit, y_encoded_fit)
        self.is_fitted = True # Set is_fitted to True after meta-learner is fitted

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_proba = self.predict_proba(X_val)

            # Ensure y_val is numpy for scikit-learn scoring
            if isinstance(y_val, (cudf.Series, cudf.DataFrame)):
                y_val_np = y_val.to_numpy()
            else:
                y_val_np = y_val

            # Ensure val_pred and val_proba are numpy for scikit-learn scoring
            if isinstance(val_pred, (cudf.Series, cudf.DataFrame)):
                val_pred_np = val_pred.to_numpy()
            else:
                val_pred_np = val_pred

            if isinstance(val_proba, (cudf.Series, cudf.DataFrame)):
                val_proba_np = val_proba.to_numpy()
            else:
                val_proba_np = val_proba

            self.history['validation_metrics'] = {
                'accuracy': accuracy_score(y_val_np, val_pred_np),
                'roc_auc': roc_auc_score(y_val_np, val_proba_np[:, 1]),
                'f1': f1_score(y_val_np, val_pred_np),
                'precision': precision_score(y_val_np, val_pred_np),
                'recall': recall_score(y_val_np, val_pred_np),
                'log_loss': log_loss(y_val_np, val_proba_np)
            }

        print("Stacking model training completed!")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities
        Args:
            X: Features to predict
        Returns:
            Probability array (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Generate meta-features (handles potential cudf internally, returns numpy)
        meta_features = self._generate_meta_feature(X, is_training=False)

        # Convert meta-features to cudf if meta-learner is cuML Logistic Regression
        if self.device == 'cuda' and self.meta_learner == 'logistic_regression':
             meta_features_predict = cudf.DataFrame(meta_features)
        else:
             meta_features_predict = meta_features # Meta-learner predicts on numpy meta-features

        # Meta-learner predicts
        proba = self.meta_learner_model.predict_proba(meta_features_predict)

        # Convert cuML output to numpy if necessary
        if isinstance(proba, (cudf.Series, cudf.DataFrame, cp.ndarray)):
             proba = proba.to_numpy() if hasattr(proba, 'to_numpy') else cp.asnumpy(proba)

        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels
        Args:
            X: Features to predict
        Returns:
            Predicted labels (numpy array)
        """
        proba = self.predict_proba(X) # predict_proba now always returns numpy
        if self.label_encoder:
            return self.label_encoder.inverse_transform(proba.argmax(axis=1))
        return proba.argmax(axis=1)

    def plot_log(self, X_test: pd.DataFrame, y_test: pd.Series,
                 save_path: str = 'plots', prefix: str = 'stacking_model') -> None:
        """
        Plot comprehensive evaluation metrics including correlation of meta-features and SHAP summary.
        Args:
            X_test: Test features
            y_test: Test target
            save_path: Directory to save plots
            prefix: Prefix for plot filenames
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        os.makedirs(save_path, exist_ok=True)

        # Predictions (will be numpy arrays)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Ensure y_test is numpy for scikit-learn plotting functions
        if isinstance(y_test, (cudf.Series, cudf.DataFrame)):
             y_test_np = y_test.to_numpy()
        else:
             y_test_np = y_test

        # Classification report
        report = classification_report(y_test_np, y_pred, output_dict=True)

        # --- Generate meta-features for plotting ---
        X_test_meta = self._generate_meta_feature(X_test, is_training=False)
        feature_names = list(self.base_estimators.keys())
        X_test_meta_df = pd.DataFrame(X_test_meta, columns=feature_names, index=X_test.index)

        # --- Create Subplots for the first 5 plots ---
        fig, axes = plt.subplots(2, 3, figsize=(20, 12)) # 2 rows, 3 columns
        fig.suptitle(f'{prefix} - Model Evaluation and Meta-Feature Analysis', fontsize=16)

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test_np, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # 2. Classification Report Heatmap
        report_df = pd.DataFrame(report).transpose()
        sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt='.3f',
                   cmap='Blues', ax=axes[1])
        axes[1].set_title('Classification Report')

        # 3. ROC Curve (for binary classification)
        if len(np.unique(y_test_np)) == 2:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test_np, y_proba[:, 1])
            auc = roc_auc_score(y_test_np, y_proba[:, 1])

            axes[2].plot(fpr, tpr, color='darkorange', lw=2,
                           label=f'ROC curve (AUC = {auc:.3f})')
            axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[2].set_xlim([0.0, 1.0])
            axes[2].set_ylim([0.0, 1.05])
            axes[2].set_xlabel('False Positive Rate')
            axes[2].set_ylabel('True Positive Rate')
            axes[2].set_title('ROC Curve')
            axes[2].legend(loc="lower right")
        else:
            axes[2].text(0.5, 0.5, 'ROC Curve\n(Binary Classification Only)',
                           ha='center', va='center', transform=axes[2].transAxes)

        # 4. Feature Importance (if meta-learner supports it)
        try:
            importances = None
            if hasattr(self.meta_learner_model, 'feature_importances_'):
                importances = self.meta_learner_model.feature_importances_
            elif hasattr(self.meta_learner_model, 'coef_'):
                coef_values = self.meta_learner_model.coef_
                if isinstance(coef_values, (cudf.Series, cudf.DataFrame, cp.ndarray)):
                    coef_values = coef_values.to_numpy() if hasattr(coef_values, 'to_numpy') else cp.asnumpy(coef_values)
                elif isinstance(coef_values, pd.Series):
                    coef_values = coef_values.to_numpy()
                importances = np.abs(coef_values[0])

            if importances is not None:
                indices = np.argsort(importances)[::-1]
                axes[3].bar(range(len(importances)), importances[indices])
                axes[3].set_xticks(range(len(importances)))
                axes[3].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                axes[3].set_title('Meta-Learner Feature Importance')
                axes[3].set_ylabel('Importance')
            else:
                axes[3].text(0.5, 0.5, 'Feature Importance\n(Not Available)',
                               ha='center', va='center', transform=axes[3].transAxes)
        except Exception as e:
            axes[3].text(0.5, 0.5, f'Feature Importance\n(Error: {e})',
                           ha='center', va='center', transform=axes[3].transAxes)

        # 5. Correlation between Base Estimator Predictions (Meta-Features)
        meta_feature_corr = X_test_meta_df.corr()
        sns.heatmap(meta_feature_corr, annot=True, cmap='coolwarm', fmt=".2f",
                    linewidths=.5, ax=axes[4])
        axes[4].set_title('Correlation of Base Estimator Predictions')
        axes[4].set_xticklabels(meta_feature_corr.columns, rotation=45, ha='right')
        axes[4].set_yticklabels(meta_feature_corr.index, rotation=0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to leave space for suptitle
        plt.savefig(os.path.join(save_path, f'{prefix}_evaluation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # --- Generate and save SHAP Summary Plot separately ---

        X_test_meta = self._generate_meta_feature(X_test, is_training=False)
        feature_names = list(self.base_estimators.keys())
        X_test_meta_df = pd.DataFrame(X_test_meta, columns=feature_names, index=X_test.index)
        
        generate_stacking_shap(self, X_test_meta_df, save_path, prefix)
        
        # Save metrics to file
        metrics = {
            'accuracy': accuracy_score(y_test_np, y_pred),
            'roc_auc': roc_auc_score(y_test_np, y_proba[:, 1]) if len(np.unique(y_test_np)) == 2 else None,
            'f1': f1_score(y_test_np, y_pred, average='weighted'),
            'precision': precision_score(y_test_np, y_pred, average='weighted'),
            'recall': recall_score(y_test_np, y_pred, average='weighted'),
            'log_loss': log_loss(y_test_np, y_proba)
        }

        with open(os.path.join(save_path, f'{prefix}_metrics.txt'), 'w') as f:
            for metric, value in metrics.items():
                if value is not None:
                    f.write(f"{metric}: {value:.4f}\n")

        print(f"Evaluation plots saved to {save_path}")

    def save_model(self, filepath: str) -> None:
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        model_data = {
            'base_estimators': self.base_estimators,
            'trained_base_estimators': self.trained_base_estimators, # Storing trained estimators might be large
            'meta_learner_model': self.meta_learner_model,
            'label_encoder': self.label_encoder,
            'history': self.history,
            'params': {
                'n_splits': self.n_splits,
                'random_state': self.random_state,
                'device': self.device,
                'meta_learner': self.meta_learner
            }
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> 'StackingModel':
        """Load a trained model"""
        model_data = joblib.load(filepath)

        self.base_estimators = model_data['base_estimators']
        self.trained_base_estimators = model_data['trained_base_estimators']
        self.meta_learner_model = model_data['meta_learner_model']
        self.label_encoder = model_data['label_encoder']
        self.history = model_data['history']

        params = model_data['params']
        self.n_splits = params['n_splits']
        self.random_state = params['random_state']
        self.device = params['device']
        self.meta_learner = params['meta_learner']

        self.is_fitted = True
        print(f"Model loaded from {filepath}")
        return self