import streamlit as st
import pandas as pd
from datetime import datetime
import sys 
import os 
import sqlite3
import base64
import requests
from ui.helper import get_input, check_api_health
from streamlit_autorefresh import st_autorefresh
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.utils import ConfigManager
import warnings
warnings.filterwarnings("ignore")
# ==============================
# PAGE CONFIG 
# ==============================
st.set_page_config(
    page_title='Credit Scoring Dashboard',
    page_icon='🌦️',
    layout='wide',
)
# ==============================
# GLOBAL CSS 
# ==============================
def load_css(file_path: str='src/ui/assets/css/style.css'):
    with open(file_path) as file:
        st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)

load_css()
# ==============================
# PATH 
# ==============================
API_URL = 'http://localhost:8000/api/v1'
CONFIG_PATH = 'config/config.yaml'
# ==============================
# CONFIG
# ==============================
cm = ConfigManager(CONFIG_PATH)
config = cm.load_config()
# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data(DATA_PATH = config['path']['raw_data']):
    return pd.read_csv(DATA_PATH)

df = load_data()
###################
@st.cache_data
def load_sqlite(DB_PATH = config['path']['database']):
    db_name = 'new_data.db'
    full_path = os.path.join(DB_PATH, db_name)
    conn = sqlite3.connect(full_path)
    df = pd.read_sql("SELECT * FROM predictions", conn)
    return df

df_sqlite = load_sqlite()
# ==============================
# CONVERT IMAGE TO BASE64
# ==============================
def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()
    
img_analyze = get_base64_image("src/ui/assets/images/analyzing.png")
img_folder = get_base64_image("src/ui/assets/images/folder.png")
img_ds = get_base64_image("src/ui/assets/images/data-science.png")
# ==============================
# AUTO REFRESH CLOCK
# ==============================
st_autorefresh(interval=1000, key="clock")
# Ssession state
if 'api_ready' not in st.session_state:
    st.session_state.api_ready = check_api_health(API_URL)
if 'warning_msg' not in st.session_state:
    st.session_state.warning_msg = None
if 'result' not in st.session_state:
    st.session_state.result = None
# ==============================
# LAYOUT
# ==============================
top_left_, top_right_ = st.columns([0.75, 0.25], gap='small')
# ==============================
# LEFT SIDE
# ==============================
with top_left_:
    # ===== METRICS =====
    # new data for metrics
    total_sql_data = len(df_sqlite)
    default_sql_data = df_sqlite[df_sqlite['prediction']==1].shape[0]
    nondefault_sql_data = total_sql_data - default_sql_data
    # print(df_sqlite)
    #
    total = len(df) + total_sql_data
    default = df[df['loan_status'] == 1].shape[0] + default_sql_data
    non_default = df[df['loan_status'] == 0].shape[0] + nondefault_sql_data
    
    col1, col2, col3 = st.columns(3)
    with col1.container(border=True, gap='xxlarge'):
        left, right = st.columns(2)
        left.metric("Total Data", f"{total:,.0f}", width='content', delta=f"{total_sql_data} Data")
        right.markdown(f'''
                       <div class='img_card'>
                       <img src="data:image/png;base64,{img_folder}">
                       </div>''', unsafe_allow_html=True)


    with col2.container(border=True):
        left, right = st.columns(2)
        left.metric("Default", f"{default:,.0f}", width='content', delta=f"{default_sql_data} Data")
        right.markdown(f'''
                       <div class='img_card'>
                       <img src="data:image/png;base64,{img_analyze}">
                       </div>''', unsafe_allow_html=True)

    with col3.container(border=True):
        left, right = st.columns(2)
        left.metric("Non Default", f"{non_default:,.0f}", width='content', delta=f"{nondefault_sql_data} Data")
        right.markdown(f'''
                       <div class='img_card'>
                       <img src="data:image/png;base64,{img_ds}">
                       </div>''', unsafe_allow_html=True)

    # ===== CHART =====
    with st.container(height=250):
        st.subheader("How Default Risk Changes with Loan-to-Income Ratio")
        df['quantile'] = pd.qcut(df['loan_percent_income'], 5)
        data_df = df.groupby('quantile')['loan_status'].mean().reset_index()

        # convert to string
        data_df['quantile'] = [
            f"{interval.left:.2f} - {interval.right:.2f}"
            for interval in data_df['quantile']
        ]
        # rename columns
        data_df.columns = ['Loan-to-Income Ratio', 'Default Rate']
        # print(data_df)
        st.data_editor(
            data_df,
            column_config={
                "Loan-to-Income Ratio": st.column_config.TextColumn(
                    width=100
                ),
                "Default Rate": st.column_config.ProgressColumn(
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                    width=800
                ),
            },
            hide_index=True,
            disabled=True
        )

# ==============================
# RIGHT SIDE 
# ==============================
with top_right_:
    now = datetime.now()
    st.markdown(f"""
                <div class="center-box">
                <div class="clock">{now.strftime("%H:%M:%S")}</div>
                <div class="date">{now.strftime("%A, %d %B %Y")}</div>
                </div>""", unsafe_allow_html=True)

# ==============================
# BOTTOM SECTION
# ==============================
with st.container(height=650, border=True):
    if not st.session_state.api_ready:
        st.warning("Model is not ready. Please check backend service.")
        st.stop()
    #
    response = None 
    #
    left_content, right_content = st.columns([0.75, 0.25], gap='small')
    # ============================== LEFT CONTENT
    with left_content:
        st.subheader("Predict Default or Non Default Customer")
        col1, col2 = st.columns(2)
        # ========== COL 1 ====================
        min_age = config.get('schema').get('person_age').get('minimum', None)
        max_age = config.get('schema').get('person_age').get('maximum', None)
        person_age = get_input(
            label="Enter Age",
            input_type="number",
            col=col1,
            min_val=min_age,
            max_val=max_age
        )

        # ==============================
        min_income = config.get('schema').get('person_income').get('minimum', None)
        max_income = config.get('schema').get('person_income').get('maximum', None)
        person_income = get_input(
            label='Enter person income',
            input_type='number',
            col=col1,
            min_val=min_income,
            max_val=max_income
        )
        # ==============================
        options_home_ownership = (
            'RENT',
            'OWN',
            'MORTGAGE',
            'OTHER'
        )
        person_home_ownership = get_input(
            label='Enter the person home ownership',
            options=options_home_ownership,
            col=col1,
            input_type='selectbox'
        )
        # ==============================
        min_emp_length = config.get('schema').get('person_emp_length').get('minimum', None)
        max_emp_length = config.get('schema').get('person_emp_length').get('maximum', None)
        person_emp_length = get_input(
            label='Enter person employment length',
            input_type='number',
            col=col1,
            min_val=min_emp_length,
            max_val=max_emp_length
        )
        # ==============================
        options_loan_intent = (
            'PERSONAL',
            'EDUCATION',
            'MEDICAL',
            'VENTURE',
            'HOMEIMPROVEMENT',
            'DEBTCONSOLIDATION'
        )
        loan_intent = get_input(
            label='Enter loan intent',
            options=options_loan_intent,
            col=col1,
            input_type='selectbox'
        )
        # ==============================
        loan_grade = col1.selectbox(
            label='Enter loan grade',
            options = (
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G'
            ),
            index=None
        )
        # ============= COL 2 =================
        min_loan_amnt = config.get('schema').get('loan_amnt').get('minimum', None)
        max_loan_amnt = config.get('schema').get('loan_amnt').get('maximum', None)
        loan_amnt = get_input(
            label='Enter loan amount',
            input_type='number',
            col=col2,
            min_val=min_loan_amnt,
            max_val=max_loan_amnt
        )
        # ==============================
        min_loan_int_rate= config.get('schema').get('loan_int_rate').get('minimum', None)
        max_loan_int_rate = config.get('schema').get('loan_int_rate').get('maximum', None)
        loan_int_rate = get_input(
            label='Enter loan interest rate',
            input_type='number',
            col=col2,
            min_val=min_loan_int_rate,
            max_val=max_loan_int_rate
        )
        # ==============================
        min_loan_percent_income= config.get('schema').get('loan_percent_income').get('minimum', None)
        max_loan_percent_income = config.get('schema').get('loan_percent_income').get('maximum', None)
        loan_percent_income = get_input(
            label='Enter loan percent income',
            input_type='number',
            col=col2,
            min_val=min_loan_percent_income,
            max_val=max_loan_percent_income
        )
        # ==============================
        option_cb_person_default_on_file = (
            'Y',
            'N'
        )
        cb_person_default_on_file = get_input(
            label='Enter cb person default on file',
            options=option_cb_person_default_on_file,
            col=col2,
            input_type='selectbox'
        )
        # ==============================
        min_cb_person_cred_hist_length = config.get('schema').get('cb_person_cred_hist_length').get('minimum', None)
        max_cb_person_cred_hist_length = config.get('schema').get('cb_person_cred_hist_length').get('maximum', None)
        cb_person_cred_hist_length = get_input(
            label='Enter cb person cred hist length',
            input_type='number',
            col=col2,
            min_val=min_cb_person_cred_hist_length,
            max_val=max_cb_person_cred_hist_length
        )
        # collect all input
        payload = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length
        }

        if st.button('Predict', use_container_width=True, type='primary'):
            # Check if values is none
            if None in payload.values():
                st.session_state.warning_msg = "Please fill all input first..."
            else:
                st.session_state.warning_msg = None
                # DOo predict
                response = requests.post(f"{API_URL}/predict", json=payload)
        # Render warning 
        if st.session_state.warning_msg:
            st.warning(st.session_state.warning_msg)

    # ============================== RIGHT CONTENT
    with right_content:
        st.subheader("Output")
        st.info("Prediction result will appear here")
        if response is not None:
            if response.status_code == 200:
                result = response.json()
                st.session_state.result = result
            else:
                st.error(f"Error: {response.text}")

        if st.session_state.result:
            st.write(st.session_state.result)

