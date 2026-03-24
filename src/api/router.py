from fastapi import APIRouter, HTTPException
from api.schema import HealthResponse, PredictionRequest, PredictionResponse
from api.service import ModelService
from utils.utils import ConfigManager

# Router API
router = APIRouter(tags=["Model"])
# Load Config
CONFIG_PATH = "config/config.yaml"
cm = ConfigManager(CONFIG_PATH)
config = cm.load_config()
# Get best threshold and best preprocessing data from config
best_threshold = config.get('best_threshold')
best_prep = config.get('best_preprocessing')
# Path model
PATH_MODEL = config['path']['best_model']
PATH_PREPROCESSING = (
    config['path']['logrob_preprocessing']
    if best_prep == 'Log + Robust Scaler'
    else config['path']['rob_preprocessing']
)
DB_PATH = config['path']['database']
# Initialize model services
service = ModelService(
    model_path=PATH_MODEL,
    prep_path=PATH_PREPROCESSING,
    db_path=DB_PATH
)
# Load model and prep and it saved to class instances
service.load_artifacts()

# home point
@router.get('/')
def root():
    return {'message': 'API is running'}

# api check model is available
@router.get('/health', response_model=HealthResponse)
def health():
    status = service.health_check()

    if not status['model_loaded']:
        raise HTTPException(status_code=404, detail='Model not loaded')

    if not status['preprocessing_loaded']:
        raise HTTPException(status_code=404, detail='Preprocessing not loaded')

    return HealthResponse(
        status='ok',
        model_loaded=status['model_loaded'],
        preprocessing_loaded=status['preprocessing_loaded']
    )

# end point for doing prediction
@router.post('/predict', response_model=PredictionResponse)
def do_prediction(data: PredictionRequest):
    if best_threshold is None:
        raise HTTPException(status_code=500, detail='Threshold not set')
    
    # Call the services for doing a prediction
    result = service.do_predict(X=data, threshold=best_threshold)

    if not result:
        raise HTTPException(status_code=500, detail='Error during prediction')
    
    return PredictionResponse(**result)