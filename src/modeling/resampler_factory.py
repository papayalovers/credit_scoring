
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTEENN

def get_resampler():
    resamplers = {
        'SMOTE': SMOTE(random_state=42),
        'SMOTEENN': SMOTEENN(random_state=42)
    }
    return resamplers