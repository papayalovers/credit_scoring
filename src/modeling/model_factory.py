from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def create_model_object():
    return [
        {"model_name": "RandomForestClassifier", "model_object": RandomForestClassifier},
        {"model_name": "KNeighborsClassifier", "model_object": KNeighborsClassifier},
        {"model_name": "XGBClassifier", "model_object": XGBClassifier},
    ]