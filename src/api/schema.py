from pydantic import BaseModel, Field
from typing import List, Union
from enum import Enum 

# Schema for enum type features
class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"
    OTHER = "OTHER"


class LoanIntent(str, Enum):
    PERSONAL = "PERSONAL"
    EDUCATION = "EDUCATION"
    MEDICAL = "MEDICAL"
    VENTURE = "VENTURE"
    HOMEIMPROVEMENT = "HOMEIMPROVEMENT"
    DEBTCONSOLIDATION = "DEBTCONSOLIDATION"


class LoanGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class DefaultOnFile(str, Enum):
    Y = "Y"
    N = "N"
    
# Response endpoint to check models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessing_loaded: bool

# Request from front
class PredictionRequest(BaseModel):
    person_age: int 
    person_income: Union[int, float] 
    person_home_ownership: HomeOwnership
    person_emp_length: float
    loan_intent: LoanIntent
    loan_grade: LoanGrade
    loan_amnt: int 
    loan_int_rate: float 
    loan_percent_income: float 
    cb_person_default_on_file: DefaultOnFile
    cb_person_cred_hist_length: int 

# Response for front
class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    risk_level: str
    messages: List[str]