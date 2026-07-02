import numpy as np
import pandas as pd
import streamlit as st
import mlflow.sklearn
import mlflow.xgboost
import joblib

st.set_page_config(
    page_title="Credit Score Classifier",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

MLFLOW_URI  = "sqlite:///mlflow.db"
RUN_ID      = "5c128381af2343e39f89ca638f191b50"
MODEL_VER   = "2"
CLASS_NAMES = ['Good', 'Poor', 'Standard']

OCCUPATION_OPTIONS = [
    'Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer',
    'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic',
    'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Unknown', 'Writer'
]
CREDIT_MIX_OPTIONS  = ['Bad', 'Good', 'Standard', 'Unknown']
PAYMENT_MIN_OPTIONS = ['No', 'Unknown', 'Yes']
PAYMENT_BEH_OPTIONS = [
    'High_spent_Large_value_payments',
    'High_spent_Medium_value_payments',
    'High_spent_Small_value_payments',
    'Low_spent_Large_value_payments',
    'Low_spent_Medium_value_payments',
    'Low_spent_Small_value_payments',
]

FEATURE_ORDER = [
    'Age', 'Occupation', 'Annual_Income', 'Monthly_Inhand_Salary',
    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt',
    'Credit_Utilization_Ratio', 'Payment_of_Min_Amount', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Payment_Behaviour', 'Monthly_Balance',
    'Credit_History_Months',
]

TEST_CASES = {
    "Good Credit": {
        "Age": 45, "Occupation": "Engineer", "Annual_Income": 120000.0,
        "Monthly_Inhand_Salary": 9500.0, "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 4, "Interest_Rate": 7, "Num_of_Loan": 2.0,
        "Delay_from_due_date": 0, "Num_of_Delayed_Payment": 0.0,
        "Num_Credit_Inquiries": 1.0, "Outstanding_Debt": 500.0,
        "Credit_Utilization_Ratio": 15.0, "Total_EMI_per_month": 200.0,
        "Amount_invested_monthly": 3000.0, "Monthly_Balance": 5000.0,
        "Credit_History_Months": 180.0, "Changed_Credit_Limit": 10.0,
        "Credit_Mix": "Good", "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "High_spent_Large_value_payments",
    },
    "Poor Credit": {
        "Age": 23, "Occupation": "Unknown", "Annual_Income": 18000.0,
        "Monthly_Inhand_Salary": 1200.0, "Num_Bank_Accounts": 8,
        "Num_Credit_Card": 9, "Interest_Rate": 34, "Num_of_Loan": 7.0,
        "Delay_from_due_date": 60, "Num_of_Delayed_Payment": 18.0,
        "Num_Credit_Inquiries": 15.0, "Outstanding_Debt": 4500.0,
        "Credit_Utilization_Ratio": 90.0, "Total_EMI_per_month": 1800.0,
        "Amount_invested_monthly": 0.0, "Monthly_Balance": 100.0,
        "Credit_History_Months": 12.0, "Changed_Credit_Limit": -5.0,
        "Credit_Mix": "Bad", "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "Low_spent_Small_value_payments",
    },
    "Standard Credit": {
        "Age": 34, "Occupation": "Teacher", "Annual_Income": 55000.0,
        "Monthly_Inhand_Salary": 4200.0, "Num_Bank_Accounts": 4,
        "Num_Credit_Card": 5, "Interest_Rate": 15, "Num_of_Loan": 3.0,
        "Delay_from_due_date": 10, "Num_of_Delayed_Payment": 5.0,
        "Num_Credit_Inquiries": 5.0, "Outstanding_Debt": 2000.0,
        "Credit_Utilization_Ratio": 45.0, "Total_EMI_per_month": 600.0,
        "Amount_invested_monthly": 800.0, "Monthly_Balance": 1500.0,
        "Credit_History_Months": 72.0, "Changed_Credit_Limit": 3.0,
        "Credit_Mix": "Standard", "Payment_of_Min_Amount": "Unknown",
        "Payment_Behaviour": "Low_spent_Medium_value_payments",
    },
}


@st.cache_resource
def load_artifacts():
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/best_model")
    except Exception:
        model = mlflow.xgboost.load_model(f"runs:/{RUN_ID}/best_model")

    scaler        = joblib.load("model_artifacts/scaler.pkl")
    encoders      = joblib.load("model_artifacts/encoders.pkl")
    impute_values = joblib.load("model_artifacts/impute_values.pkl")
    num_cols      = joblib.load("model_artifacts/num_cols.pkl")

    return model, scaler, encoders, impute_values, num_cols


def preprocess(inputs, scaler, encoders, impute_values, num_cols):
    df = pd.DataFrame([inputs.copy()])

    for col, val in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).apply(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    df[num_cols] = scaler.transform(df[num_cols])
    return df[FEATURE_ORDER]


def predict(model, inputs, scaler, encoders, impute_values, num_cols):
    df       = preprocess(inputs, scaler, encoders, impute_values, num_cols)
    pred_idx = model.predict(df)[0]
    label    = CLASS_NAMES[pred_idx]
    proba    = model.predict_proba(df)[0] if hasattr(model, 'predict_proba') else [0, 0, 0]
    confidence = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}
    return label, confidence


if 'form_values' not in st.session_state:
    st.session_state.form_values = {}

with st.sidebar:
    st.header("Credit Score Classifier")
    st.caption(f"Model: Random Forest")
    st.divider()

    st.subheader("Test Cases")
    for case_name, case_values in TEST_CASES.items():
        if st.button(case_name, key=f"btn_{case_name}", use_container_width=True):
            st.session_state.form_values = case_values.copy()
            st.rerun()

    st.divider()


st.title("Credit Score Classifier")
st.caption("Machine Learning · Model Deployment · Random Forest")
st.divider()

col_form, col_result = st.columns([1.4, 1], gap="large")
fv = st.session_state.form_values

with col_form:
    st.subheader("Customer Data Input")

    st.markdown("**Personal Information**")
    c1, c2 = st.columns(2)
    age        = c1.number_input("Age", min_value=18, max_value=100, value=int(fv.get('Age', 30)))
    occupation = c2.selectbox("Occupation", OCCUPATION_OPTIONS,
                               index=OCCUPATION_OPTIONS.index(fv.get('Occupation', 'Engineer'))
                               if fv.get('Occupation') in OCCUPATION_OPTIONS else 0)

    st.markdown("**Income & Balance**")
    c1, c2, c3 = st.columns(3)
    annual_income   = c1.number_input("Annual Income",   value=float(fv.get('Annual_Income', 50000)), step=1000.0)
    monthly_salary  = c2.number_input("Monthly Salary",  value=float(fv.get('Monthly_Inhand_Salary', 4000)), step=100.0)
    monthly_balance = c3.number_input("Monthly Balance", value=float(fv.get('Monthly_Balance', 1000)), step=100.0)

    st.markdown("**Credit Profile**")
    c1, c2, c3 = st.columns(3)
    num_bank_accounts     = c1.number_input("Bank Accounts",   min_value=0, max_value=20, value=int(fv.get('Num_Bank_Accounts', 3)))
    num_credit_card       = c2.number_input("Credit Cards",    min_value=0, max_value=20, value=int(fv.get('Num_Credit_Card', 4)))
    num_of_loan           = c3.number_input("Number of Loans", min_value=0, max_value=20, value=int(fv.get('Num_of_Loan', 2)))

    c1, c2, c3 = st.columns(3)
    interest_rate         = c1.number_input("Interest Rate (%)",       min_value=0, max_value=60,      value=int(fv.get('Interest_Rate', 10)))
    credit_utilization    = c2.number_input("Credit Utilization (%)",  min_value=0.0, max_value=100.0, value=float(fv.get('Credit_Utilization_Ratio', 30)))
    credit_history_months = c3.number_input("Credit History (months)", min_value=0, max_value=600,     value=int(fv.get('Credit_History_Months', 60)))

    c1, c2 = st.columns(2)
    outstanding_debt     = c1.number_input("Outstanding Debt",     min_value=0.0, value=float(fv.get('Outstanding_Debt', 1000)), step=100.0)
    changed_credit_limit = c2.number_input("Changed Credit Limit", value=float(fv.get('Changed_Credit_Limit', 5)), step=1.0)

    st.markdown("**Payment Behavior**")
    c1, c2 = st.columns(2)
    delay_from_due      = c1.number_input("Delay from Due Date (days)", min_value=0, value=int(fv.get('Delay_from_due_date', 5)))
    num_delayed_payment = c2.number_input("Num of Delayed Payments",    min_value=0, value=int(fv.get('Num_of_Delayed_Payment', 3)))

    c1, c2, c3 = st.columns(3)
    credit_mix_cat    = c1.selectbox("Credit Mix",        CREDIT_MIX_OPTIONS,
                                      index=CREDIT_MIX_OPTIONS.index(fv.get('Credit_Mix', 'Standard'))
                                      if fv.get('Credit_Mix') in CREDIT_MIX_OPTIONS else 2)
    payment_min       = c2.selectbox("Pays Min Amount",   PAYMENT_MIN_OPTIONS,
                                      index=PAYMENT_MIN_OPTIONS.index(fv.get('Payment_of_Min_Amount', 'No'))
                                      if fv.get('Payment_of_Min_Amount') in PAYMENT_MIN_OPTIONS else 0)
    payment_behaviour = c3.selectbox("Payment Behaviour", PAYMENT_BEH_OPTIONS,
                                      index=PAYMENT_BEH_OPTIONS.index(fv.get('Payment_Behaviour', PAYMENT_BEH_OPTIONS[0]))
                                      if fv.get('Payment_Behaviour') in PAYMENT_BEH_OPTIONS else 0)

    st.markdown("**EMI & Investment**")
    c1, c2, c3 = st.columns(3)
    total_emi            = c1.number_input("Total EMI/month",       min_value=0.0, value=float(fv.get('Total_EMI_per_month', 300)), step=50.0)
    amount_invested      = c2.number_input("Amount Invested/month", min_value=0.0, value=float(fv.get('Amount_invested_monthly', 500)), step=50.0)
    num_credit_inquiries = c3.number_input("Credit Inquiries",      min_value=0,   value=int(fv.get('Num_Credit_Inquiries', 3)))

    st.write("")
    predict_btn = st.button("Predict Credit Score", type="primary", use_container_width=True)


with col_result:
    st.subheader("Prediction Result")

    if predict_btn:
        model, scaler, encoders, impute_values, num_cols = load_artifacts()

        inputs = {
            'Age': age, 'Occupation': occupation,
            'Annual_Income': annual_income,
            'Monthly_Inhand_Salary': monthly_salary,
            'Num_Bank_Accounts': num_bank_accounts,
            'Num_Credit_Card': num_credit_card,
            'Interest_Rate': interest_rate,
            'Num_of_Loan': num_of_loan,
            'Delay_from_due_date': delay_from_due,
            'Num_of_Delayed_Payment': num_delayed_payment,
            'Changed_Credit_Limit': changed_credit_limit,
            'Num_Credit_Inquiries': num_credit_inquiries,
            'Credit_Mix': credit_mix_cat,
            'Outstanding_Debt': outstanding_debt,
            'Credit_Utilization_Ratio': credit_utilization,
            'Payment_of_Min_Amount': payment_min,
            'Total_EMI_per_month': total_emi,
            'Amount_invested_monthly': amount_invested,
            'Payment_Behaviour': payment_behaviour,
            'Monthly_Balance': monthly_balance,
            'Credit_History_Months': credit_history_months,
        }

        with st.spinner("Analyzing..."):
            label, confidence = predict(model, inputs, scaler, encoders, impute_values, num_cols)

        desc = {
            "Good":     "Low risk customer. Excellent credit history.",
            "Standard": "Medium risk customer. Average credit behavior.",
            "Poor":     "High risk customer. Multiple credit issues detected.",
        }.get(label, "")

        if label == "Good":
            st.success(f"**{label}** — {desc}")
        elif label == "Standard":
            st.warning(f"**{label}** — {desc}")
        else:
            st.error(f"**{label}** — {desc}")

        st.markdown("**Confidence Scores**")
        for cls in ["Good", "Standard", "Poor"]:
            pct = confidence.get(cls, 0.0)
            st.progress(pct, text=f"{cls}: {pct*100:.1f}%")

        st.markdown("**Input Summary**")
        summary_df = pd.DataFrame({
            "Feature": ["Age", "Annual Income", "Outstanding Debt",
                        "Delayed Payments", "Credit History", "Credit Mix"],
            "Value": [
                age, f"${annual_income:,.0f}", f"${outstanding_debt:,.0f}",
                num_delayed_payment, f"{credit_history_months} months", credit_mix_cat
            ]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    else:
        st.info("You can fill  customer data on the left, or load a test case from the sidebar, then click Predict Credit Score to showsase the score.")