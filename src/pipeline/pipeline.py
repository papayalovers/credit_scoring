from utils.utils import ConfigManager, BASE_DIR
from data.data_preparation import DataPreparation
from features.preprocessing import Preprocessing
import logging 
from modeling.resampler_factory import get_resampler
from modeling.trainer_model import TrainerModel
from utils.logger import _update_training_log
from modeling.model_factory import create_model_object
from modeling.evaluation import auto_selector_treshold, evaluation_on_test_set

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/config.yaml"


def run_training_pipeline(
    progress_callback=None,
    step_done_callback=None
):

    def update(step, msg, inc=1):
        if progress_callback:
            progress_callback(step, msg, inc)

    def step_done(step):
        if step_done_callback:
            step_done_callback(step)

    # Load configuration
    cm = ConfigManager(CONFIG_PATH)
    config = cm.load_config()

    try:
        logger.info("Starting training pipeline...")
        ###########################
        # 1. Do data preparation
        ###########################
        try:
            logger.info("========== STEP: DATA PREPARATION ==========")
            step_name = "Data Preparation"

            update(step_name, "Loading config...", 5)
            # Initialize DataPreparation with config
            data_prep = DataPreparation(config)
            # Load raw data
            update(step_name, "Loading raw data...", 5)
            raw_data = data_prep.load_raw_data()

            # Do data validation
            update(step_name, "Validating data...", 10)
            is_valid = data_prep.validate_data(raw_data)

            # If data is not valid, do cleaning
            if not is_valid:
                update(step_name, "Cleaning data...", 10)
                cleaned_df = data_prep.clean_data(raw_data)
            else:
                cleaned_df = raw_data
            logger.info(f"Data after cleaning {cleaned_df.shape}")

            # Do data defense
            update(step_name, "Running data defense...", 10)
            data_prep.data_defense(cleaned_df)

            # Split input and output
            update(step_name, "Splitting features & target...")
            X, y = data_prep.split_input_output(cleaned_df)

            # Split into train, valid, test
            update(step_name, "Train-test split...", 20)
            X_train, X_not_test, y_train, y_not_test = (
                data_prep.split_x_y(X, y, test_size=0.4)
            )

            X_valid, X_test, y_valid, y_test = (
                data_prep.split_x_y(X_not_test, y_not_test, test_size=0.5)
            )
            logger.info(f"Shape X_train: {X_train.shape}, y_train: {y_train.shape}")
            logger.info(f"Shape X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
            logger.info(f"Shape X_test: {X_test.shape}, y_test: {y_test.shape}")
            logger.info(
                f"Composition of Data Train, Valid and Test: "
                f"{X_train.shape[0]/len(cleaned_df):.0%} - "
                f"{X_valid.shape[0]/len(cleaned_df):.0%} - "
                f"{X_test.shape[0]/len(cleaned_df):.0%}"
            )            
            logger.info(f"All split composition {sum([len(X_train), len(X_valid), len(X_test)])}")
            # Serialize the data 
            update(step_name, "Serializing data...", 20)
            train = {"X_train": X_train, "y_train": y_train}
            valid = {"X_valid": X_valid, "y_valid": y_valid}
            test = {"X_test": X_test, "y_test": y_test}

            PATH_INTERIM = config.get('path', {}).get('interim_data', '')

            path_train = f"{PATH_INTERIM}/train.pkl"
            path_valid = f"{PATH_INTERIM}/valid.pkl"
            path_test = f"{PATH_INTERIM}/test.pkl"

            cm.serialized_data(train, path_train)
            cm.serialized_data(valid, path_valid)
            cm.serialized_data(test, path_test)

            # Update Config
            update(step_name, "Updating config...", 20)
            cm.update_config('path.train_data', path_train)
            cm.update_config('path.valid_data', path_valid)
            cm.update_config('path.test_data', path_test)

            step_done(step_name)
            logger.info("Data preparation completed successfully.")
        except Exception as e:
            logger.exception(f"Error during data preparation: {e}")
            raise
        ###########################
        # 2. Do data preprocessing
        ###########################
        try:
            logger.info("========== STEP: DATA PREPROCESSING ==========")
            step_name = "Data Preprocessing"

            # Load path
            update(step_name, "Loading dataset paths...", 5)
            train_data_path = config['path']['train_data']
            valid_data_path = config['path']['valid_data']
            test_data_path = config['path']['test_data']

            # Deserialize
            update(step_name, "Deserializing datasets...", 10)
            data_train = cm.deserialize_data(train_data_path)
            data_valid = cm.deserialize_data(valid_data_path)
            data_test = cm.deserialize_data(test_data_path)

            logger.info("Datasets successfully loaded from disk")

            # Split X, y
            update(step_name, "Splitting features and target...", 5)
            X_train, y_train = data_train['X_train'], data_train['y_train']
            X_valid, y_valid = data_valid['X_valid'], data_valid['y_valid']
            X_test, y_test = data_test['X_test'], data_test['y_test']

            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_valid shape: {X_valid.shape}")
            logger.info(f"X_test shape: {X_test.shape}")

            # ------------------------
            # Robust Scaler Pipeline
            # ------------------------
            update(step_name, "Initializing Robust Scaler pipeline...", 10)

            rob_preprocessing = Preprocessing(config=config, is_log=False).build_pipeline()
            rob_preprocessing.set_output(transform='pandas')

            update(step_name, "Fitting Robust Scaler pipeline...", 10)
            X_train_rob = rob_preprocessing.fit_transform(X_train)

            update(step_name, "Transforming validation data (Robust)...", 5)
            X_valid_rob = rob_preprocessing.transform(X_valid)

            update(step_name, "Transforming test data (Robust)...", 5)
            X_test_rob = rob_preprocessing.transform(X_test)

            logger.info(f"Robust Scaler output shape: {X_train_rob.shape}")

            # ------------------------
            # Log + Robust Pipeline
            # ------------------------
            update(step_name, "Initializing Log + Robust pipeline...", 10)

            logrob_preprocessing = Preprocessing(config=config, is_log=True).build_pipeline()
            logrob_preprocessing.set_output(transform='pandas')

            update(step_name, "Fitting Log + Robust pipeline...", 10)
            X_train_logrob = logrob_preprocessing.fit_transform(X_train)

            update(step_name, "Transforming validation data (Log + Robust)...", 5)
            X_valid_logrob = logrob_preprocessing.transform(X_valid)

            update(step_name, "Transforming test data (Log + Robust)...", 5)
            X_test_logrob = logrob_preprocessing.transform(X_test)

            logger.info(f"Log + Robust output shape: {X_train_logrob.shape}")

            # ------------------------
            # Serialization
            update(step_name, "Preparing processed datasets...", 5)
            data_train = {
                'Robust Scaler': {
                    'X_train': X_train_rob,
                    'y_train': y_train
                },
                'Log + Robust Scaler': {
                    'X_train': X_train_logrob,
                    'y_train': y_train
                }
            }
            data_valid = {
                'Robust Scaler': {
                    'X_valid': X_valid_rob,
                    'y_valid': y_valid
                },
                'Log + Robust Scaler': {
                    'X_valid': X_valid_logrob,
                    'y_valid': y_valid
                }
            }
            data_test = {
                'Robust Scaler': {
                    'X_test': X_test_rob,
                    'y_test': y_test
                },
                'Log + Robust Scaler': {
                    'X_test': X_test_logrob,
                    'y_test': y_test
                }
            }
            # Serialized Data
            update(step_name, "Saving processed datasets...", 15)
            PATH_PROCESSED = config['path']['processed_data']
            path_train = f"{PATH_PROCESSED}/train_pipeline.pkl"
            path_valid = f"{PATH_PROCESSED}/valid_pipeline.pkl"
            path_test = f"{PATH_PROCESSED}/test_pipeline.pkl"
            
            cm.serialized_data(data_train, path_train)
            cm.serialized_data(data_valid, path_valid)
            cm.serialized_data(data_test, path_test)
            # Serialize Preprocessing pipeline
            PATH_PROCESSING_PIPELINE = config['path']['models']
            path_rob_processing = f"{PATH_PROCESSING_PIPELINE}/rob_prep_pipeline.pkl"
            path_logrob_processing = f"{PATH_PROCESSING_PIPELINE}/logrob_prep_pipeline.pkl"

            cm.serialized_data(rob_preprocessing, path_rob_processing)
            cm.serialized_data(logrob_preprocessing, path_logrob_processing)
            # Update Config
            cm.update_config('path.train_data_processed_pipe', path_train)
            cm.update_config('path.valid_data_processed_pipe', path_valid)
            cm.update_config('path.test_data_processed_pipe', path_test)
            cm.update_config('path.rob_preprocessing', path_rob_processing)
            cm.update_config('path.logrob_preprocessing', path_logrob_processing)
            logger.info("Processed datasets successfully saved")
            step_done(step_name)
            logger.info("Data preprocessing completed successfully.")
        except Exception as e:
            logger.exception(f"Error during data preprocessing: {e}")
            raise
        # =========================
        # 3. Do training model
        # =========================
        try:
            logger.info("========== STEP: TRAINING MODEL ==========")
            step_name = "Training Model" 
            # Load Train Data
            update(step_name, "Load train dataset...", 5)
            data_train = cm.deserialize_data(path_train)
            # Load resampler factory
            update(step_name, "Load resampler factory...", 5)
            resamplers = get_resampler()
            # Estimate total training combinations
            n_prep = len(data_train)
            n_res = len(resamplers)
            temp_models = create_model_object()
            n_models = len(temp_models)
            #
            total_runs = n_prep * n_res * n_models
            update(step_name, f"Total experiments: {total_runs}", 5)
            # Initialize Trainer Models
            trainer = TrainerModel(
                n_trials=15,
                cv=5,
                scoring='Recall',
                progress_callback=update
            )
            # Start training
            update(step_name, "Start model training...", 10)
            training_log = trainer.train(data=data_train, resamplers=resamplers)

            # Save log training
            update(step_name, "Saving training logs...", 10)
            PATH_TRAINING_LOG = config['path']['experiments']
            _ = _update_training_log(training_log, PATH_TRAINING_LOG)

            # Get best model
            update(step_name, "Selecting best model...", 5)
            best_row = trainer.get_best_model()

            # Rebuild pipeline
            update(step_name, "Rebuilding best pipeline...", 5)
            model_pipeline, prep_name = trainer.rebuild_pipeline(
                best_row=best_row,
                resamplers=resamplers
            )

            # Retrain best model
            update(step_name, f"Fitting best model ({best_row})...", 10)
            best_model = trainer.fit_best_model(
                pipeline=model_pipeline,
                prep_name=prep_name,
                data=data_train
            )

            # Save best model
            update(step_name, f"Saving best model...", 10)
            PATH_MODEL = config['path']['models']
            model_path = f"{PATH_MODEL}/best_model.pkl"

            cm.serialized_data(best_model, model_path)
            cm.update_config('path.best_model', model_path)
            # Save preprocessing info
            cm.update_config('best_preprocessing', best_row['preprocessing'])

            step_done(step_name)
            logger.info(f"Training model Completed Successfully")
        except Exception as e:
            logger.exception(f"Error during training models: {e}")
        # =========================
        # 4. Evaluation Model
        # =========================
        try:
            logger.info("========== STEP: EVALUATION MODEL ==========")
            step_name = "Evaluation Model" 

            # Load best model
            update(step_name, "Loading best model...", 5)
            PATH_BEST_MODEL = config['path']['best_model']
            best_model = cm.deserialize_data(PATH_BEST_MODEL)

            # Load dataset paths
            update(step_name, "Loading dataset paths...", 5)
            valid_data_path = config['path']['valid_data_processed_pipe']
            test_data_path = config['path']['test_data_processed_pipe']

            # Deserialize datasets
            update(step_name, "Deserializing validation & test data...", 10)
            data_valid = cm.deserialize_data(valid_data_path)
            data_test = cm.deserialize_data(test_data_path)

            # Load best preprocessing
            update(step_name, "Loading best preprocessing config...", 5)
            best_prep = config['best_preprocessing']

            # Split X, y
            update(step_name, "Preparing validation & test features...", 10)
            X_valid = data_valid[best_prep]['X_valid']
            y_valid = data_valid[best_prep]['y_valid']

            X_test = data_test[best_prep]['X_test']
            y_test = data_test[best_prep]['y_test']

            # Threshold tuning
            update(step_name, "Running threshold tuning (validation set)...", 20)
            best_threshold, _ = auto_selector_treshold(
                model=best_model,
                X_valid=X_valid,
                y_valid=y_valid,
                min_precision=0.5,
            )

            # Evaluation on test set
            update(step_name, "Evaluating model on test set...", 25)
            results = evaluation_on_test_set(
                best_model=best_model,
                best_threshold=best_threshold,
                X_test=X_test,
                y_test=y_test
            )

            # Save results
            update(step_name, "Saving threshold & evaluation results...", 10)
            cm.update_config('best_threshold', float(best_threshold))
            cm.update_config('test_set_eval_result', results)

            step_done(step_name)
            logger.info(f"Evaluation step completed successfully - result: {results}.")
        except Exception as e:
            logger.exception(f"Error during evaluation model: {e}")
    except Exception as e:
        logger.exception(f"Error in training pipeline: {e}")
        raise

    