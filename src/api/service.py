import joblib
import os
import pandas as pd
from enum import Enum

class ModelService:
    def __init__(self, model_path: str, prep_path: str, db_path: str):
        self.model_path = model_path
        self.prep_path = prep_path
        self.db_path = db_path
        self.model = None
        self.preprocessing = None

    def _save_prediction(self, data: pd.DataFrame, result: dict) -> None:
        import sqlite3 
        
        db_name = 'new_data.db'
        full_path = os.path.join(self.db_path, db_name)
        
        conn = sqlite3.connect(full_path)
        cursor = conn.cursor()

        cursor.execute(''' 
        INSERT INTO predictions (
            person_age,
            person_income,
            person_home_ownership,
            person_emp_length,
            loan_intent,
            loan_grade,
            loan_amnt,
            loan_int_rate,
            loan_percent_income,
            cb_person_default_on_file,
            cb_person_cred_hist_length,
            probability, 
            prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(data['person_age'].iloc[0]),
            float(data['person_income'].iloc[0]),
            str(data['person_home_ownership'].iloc[0]),
            float(data['person_emp_length'].iloc[0]),
            str(data['loan_intent'].iloc[0]),
            str(data['loan_grade'].iloc[0]),
            int(data['loan_amnt'].iloc[0]),
            float(data['loan_int_rate'].iloc[0]),
            float(data['loan_percent_income'].iloc[0]),
            str(data['cb_person_default_on_file'].iloc[0]),
            str(data['cb_person_cred_hist_length'].iloc[0]),
            float(result['probability']),
            float(result['prediction']),
        ))

        conn.commit()
        conn.close()

    def load_artifacts(self):
        # Load model
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)


        # Load preprocessing
        if os.path.exists(self.prep_path):
            self.preprocessing = joblib.load(self.prep_path)

        return self.model, self.preprocessing

    def health_check(self):
        return {
            "model_loaded": self.model is not None,
            "preprocessing_loaded": self.preprocessing is not None
        }
    
    def do_predict(self, X: dict, threshold: float) -> dict:
        # print(X)
        # convert Pydantic → dict
        X_dict = X.model_dump()

        # clean Enum
        X_clean = {
            k: (v.value if isinstance(v, Enum) else v)
            for k, v in X_dict.items()
        }

        # 
        data = pd.DataFrame([X_clean])
        print(data)
        X_ = self.preprocessing.transform(data)

        proba = self.model.predict_proba(X_)[:, 1]
        proba = float(proba[0])
        pred = int(proba >= threshold)
        #
        messages = list()
        # Rule business for loan percent income base on data exploration
        loan_ratio = data['loan_percent_income'].iloc[0]

        if loan_ratio > 0.25:
            messages.append(
                "Loan-to-income ratio is high (>25%). Based on historical data, customers in this range have a higher probability of default (~52%)."
            )        
        if proba > 0.7:
            risk_level = 'High'
            messages.append('High risk of default')
        elif proba > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Get the result of predictions
        result = {
            'prediction' : pred,
            'label' : 'Default' if pred == 1 else 'Non Default',
            'probability' : proba,
            'risk_level' : risk_level,
            'messages' : messages,
        }
        # Save new input data to database
        self._save_prediction(data=data, result=result)
        return result
