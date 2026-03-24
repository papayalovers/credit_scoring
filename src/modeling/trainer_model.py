import optuna
import time 
import pandas as pd
from datetime import datetime
from modeling.model_factory import create_model_object
from modeling.param_spaces import get_model_and_params
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from typing import Union
import numpy as np
import logging 

logger = logging.getLogger(__name__)
# suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

class TrainerModel:
    '''
    Class for training multiple ML models with resampling and hyperparameter tuning (Optuna).

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials per model
    cv : int
        Cross-validation folds
    scoring : str
        Evaluation metric (default: recall)
    '''
    def __init__(
            self, 
            n_trials: int=20, 
            cv: int=5, 
            scoring='recall', 
            progress_callback: any=None
    ):
        self.n_trials = n_trials 
        self.cv = cv
        self.scoring = scoring.lower()
        self.experiments_log = list()
        self.models = list()
        self.progress_callback = progress_callback
    # Update PRogress 
    def _update_progress(self, message: str, inc: int = 1):
        if self.progress_callback:
            self.progress_callback("Training Model", message, inc)

    # Function to create objective
    def _create_objective(
            self, 
            X_train: pd.DataFrame, 
            y_train: Union[pd.Series, np.ndarray], 
            resampler: object, 
            model_name: str, 
            model_obj: object
    ) -> object:
        '''Create Optuna objective function'''
        def objective(trial):
            try:
                # Get the model names and parameters
                params = get_model_and_params(trial, model_name)
                model = model_obj(**params)

                pipeline = Pipeline([
                    ('resampler', resampler),
                    ('model', model)
                ])

                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=self.cv,
                    scoring=self.scoring,
                    n_jobs=-1,
                    error_score='raise'  
                )

                return scores.mean()

            except Exception as e:
                logger.error(f"[ERROR][{model_name}] Trial failed: {e}")
                raise

        return objective
    
    def train(self, data: dict, resamplers: object) -> dict:
        '''Run training for multiple preprocessing, resamplers, and models.'''
        # Basic validation
        if not data:
            raise ValueError('Input data is empty!')
        if not resamplers:
            raise ValueError('Resampler dictionary is empty!')
        
        # Get list of models
        if not self.models:
            self.models = create_model_object()
        
        self.experiments_log = []

        for prep_name, data_train in  data.items():
            # defensive
            if 'X_train' not in data_train or 'y_train' not in data_train:
                raise KeyError(f"{prep_name} missing X_train or y_train")
            # Extract X_train and y_train
            X_train = data_train['X_train']
            y_train = data_train['y_train']
            for res_name, resampler in resamplers.items():
                for model in self.models:
                    try:
                        model_name = model['model_name']
                        model_obj = model['model_object']
                        #--------------------
                        self._update_progress(f"Training: {prep_name} | {res_name} | {model_name}")
                        logger.info(f"Training: {prep_name} | {res_name} | {model_name}")
                        #--------------------
                        # Create objective function for optuna
                        objective = self._create_objective(
                            X_train,
                            y_train,
                            resampler,
                            model_name,
                            model_obj
                        )
                        # Optuna
                        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
                        start_time = time.time()
                        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
                        finish_time = time.time()

                        # 
                        training_time = finish_time - start_time
                        
                        # Training Log Result
                        log_entry = {
                            'timestamp' : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'preprocessing' : prep_name,
                            'resampler' : res_name,
                            'model' : model_name,
                            f'{self.scoring}_score' : study.best_value,
                            'params' : study.best_params,
                            'training_time' : training_time
                        }
                        #--------------------
                        self._update_progress(f"[RESULT] {prep_name} | {res_name} | {model_name} Score: {study.best_value:.4f}")
                        logger.info(f"[RESULT] {prep_name} | {res_name} | {model_name} Score: {study.best_value:.4f}")
                        #--------------------
                        self.experiments_log.append(log_entry)
                    except Exception as e:
                        logger.error(f'[ERROR] {prep_name} | {res_name} | {model_name} - {e}')

        return self.experiments_log
    
    def get_best_model(self, metric: str = 'recall_score') -> dict:
        """
        Get best model configuration from training log.

        Parameters
        ----------
        experiments_log : list
            List of experiment results
        metric : str
            Metric to optimize (default: recall_score)

        Returns
        -------
        dict
            Best experiment row
        """

        if not self.experiments_log:
            raise ValueError("Training log is empty!")

        best_row = max(self.experiments_log, key=lambda x: x[metric])
        
        logger.info(f"Best Model Found:")
        logger.info(f"Preprocessing: {best_row['preprocessing']}")
        logger.info(f"Model: {best_row['model']}")
        logger.info(f"Resampler: {best_row['resampler']}")
        logger.info(f"{metric}: {best_row[metric]:.4f}")

        return best_row
    
    def rebuild_pipeline(self, best_row: dict, resamplers: dict) -> tuple:
        """
        Rebuild pipeline from best experiment log.
        """

        model_name = best_row['model']
        resampler_name = best_row['resampler']
        params = best_row['params']
        prep_name = best_row['preprocessing']

        # Model
        model_class = None
        for m in self.models:
            if m['model_name'] == model_name:
                model_class = m['model_object']
                break

        if model_class is None:
            raise ValueError(f"Model {model_name} not supported")

        best_model = model_class(**params)

        # Resampler
        if resampler_name not in resamplers:
            raise ValueError(f"Resampler {resampler_name} not supported")
        resampler = resamplers[resampler_name]

        pipeline = Pipeline([
            ('resampler', resampler),
            ('model', best_model)
        ])

        logger.info("Pipeline successfully rebuilt from best log")

        return pipeline, prep_name
    
    def fit_best_model(
        self,
        pipeline: Pipeline,
        prep_name: str,
        data: dict
    ) -> Pipeline:
        """
        Fit best pipeline using selected preprocessing data.
        """

        if prep_name not in data:
            raise ValueError(f"Preprocessing '{prep_name}' not found in data")

        X_train = data[prep_name]['X_train']
        y_train = data[prep_name]['y_train']
        #--------------------
        logger.info(f"Fitting best model using preprocessing: {prep_name}")
        #--------------------
        pipeline.fit(X_train, y_train)
        #--------------------
        logger.info("Best model successfully trained")
        #--------------------
        return pipeline
    

