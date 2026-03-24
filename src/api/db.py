import sqlite3
import os

def init_db(db_path: str):
    db_name = 'new_data.db'
    full_path = os.path.join(db_path, db_name)

    conn = sqlite3.connect(full_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_age INTEGER NOT NULL,
        person_income INTEGER,
        person_home_ownership TEXT CHECK(person_home_ownership IN ('RENT', 'OWN', 'MORTGAGE', 'OTHER')),
        person_emp_length FLOAT,
        loan_intent TEXT CHECK(loan_intent IN ('EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT')),
        loan_grade TEXT CHECK(loan_grade IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')),
        loan_amnt INTEGER,
        loan_int_rate FLOAT,
        loan_percent_income FLOAT,
        cb_person_default_on_file TEXT CHECK(cb_person_default_on_file IN ('Y', 'N')),
        cb_person_cred_hist_length INTEGER,
        probability FLOAT,
        prediction FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    ''')

    conn.commit()
    conn.close()