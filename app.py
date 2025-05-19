import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import OrderedDict  # To maintain column order

# --- Configuration ---
# IMPORTANT: Update these paths and values based on your v2.4 training script's output!
# 1. MODEL_PATH: Should be the filename of the model saved by bank_marketing_ml_pipeline_v2.py (v2.4)
#    Example: If XGBoost was best, it would be 'XGBoost_pipeline_v2_4.pkl'
MODEL_PATH = 'XGBoost_pipeline_v2_4.pkl'  # <<< UPDATE THIS PATH/FILENAME

# 2. OPTIMAL_THRESHOLD: This should come from your training pipeline's output
#    (e.g., from section 8 of v4, or re-calculate for the best model of v2.4 if needed).
#    This is the threshold for converting probabilities to binary predictions (0 or 1)
#    that optimizes a chosen metric like F1-score for the positive class.
OPTIMAL_THRESHOLD = 0.30  # <<< UPDATE THIS with the actual optimal threshold from your training


# --- Load Model ---
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained pipeline."""
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(
            f"Error: Model file not found at '{model_path}'. Please ensure the path and filename are correct and the model was saved by the v2.4 training script.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None


# --- Feature Engineering Function ---
# This function must replicate the feature engineering steps from your training pipeline (v2.4)
# that occur *before* the ColumnTransformer.
def engineer_features_for_app(input_df_raw):
    """
    Applies feature engineering to the raw input DataFrame.
    Matches the steps in the training pipeline (v2.4) before preprocessing.
    Does NOT perform rare category consolidation here; relies on loaded pipeline's OHE.
    """
    df = input_df_raw.copy()

    # 1. 'pdays' and 'not_previously_contacted'
    if 'pdays' in df.columns:
        df['not_previously_contacted'] = (df['pdays'] == 999).astype(int)
        df.loc[df['pdays'] == 999, 'pdays'] = -1
    else:
        df['not_previously_contacted'] = 0  # Default if pdays somehow missing
        df['pdays'] = -1

    # 2. 'age_group'
    if 'age' in df.columns:
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['Young', 'Adult', 'Mid-Adult', 'Senior', 'Elderly']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False).astype(
            str)  # Ensure string for consistency
    else:
        df['age_group'] = 'Unknown'

    # 3. 'campaign_binned'
    if 'campaign' in df.columns:
        campaign_bins = [0, 1, 2, 3, 5, 10, 100]
        campaign_labels = ['1_contact', '2_contacts', '3_contacts', '4-5_contacts', '6-10_contacts', '>10_contacts']
        df['campaign_binned'] = pd.cut(df['campaign'], bins=campaign_bins, labels=campaign_labels, right=True,
                                       include_lowest=True)
        df['campaign_binned'] = df['campaign_binned'].astype(object).fillna('Unknown_Campaign_Bin').astype(str)
    else:
        df['campaign_binned'] = 'Unknown_Campaign_Bin'

    # 4. 'job_education'
    if 'job' in df.columns and 'education' in df.columns:
        df['job_education'] = df['job'].astype(str) + "_" + df['education'].astype(str)
    else:
        df['job_education'] = 'Unknown_Unknown'

    # 5. 'euribor_emp_rate_diff' (New in v2.2+)
    if 'euribor3m' in df.columns and 'emp.var.rate' in df.columns:
        df['euribor_emp_rate_diff'] = df['euribor3m'] - df['emp.var.rate']
    else:
        df['euribor_emp_rate_diff'] = 0  # Default if source columns missing

    # 6. 'poutcome_x_job' (New in v2.2+)
    if 'poutcome' in df.columns and 'job' in df.columns:
        df['poutcome_x_job'] = df['poutcome'].astype(str) + "_" + df['job'].astype(str)
    else:
        df['poutcome_x_job'] = 'Unknown_Unknown'

    return df


# --- Main Application ---
def main():
    st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")
    st.title("🏦 Bank Term Deposit Subscription Prediction (v2.4 Pipeline)")
    st.markdown("""
        This app predicts whether a customer will subscribe to a term deposit.
        Provide the customer's details below.
        **Important**: Ensure `MODEL_PATH` and `OPTIMAL_THRESHOLD` constants in the script are correctly set based on your training results from the **v2.4 pipeline**.
    """)

    pipeline = load_model(MODEL_PATH)
    if pipeline is None:
        return

    st.sidebar.header("Customer Input Features:")

    # Define options for categorical features (from original dataset)
    job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed',
                   'services', 'student', 'technician', 'unemployed', 'unknown']
    marital_options = ['divorced', 'married', 'single', 'unknown']
    education_options = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course',
                         'university.degree', 'unknown']
    default_options = ['no', 'unknown', 'yes']  # 'unknown' is a valid category
    housing_options = ['no', 'unknown', 'yes']
    loan_options = ['no', 'unknown', 'yes']
    contact_options = ['cellular', 'telephone']
    poutcome_options = ['failure', 'nonexistent', 'success', 'unknown']  # 'unknown' can be a poutcome if not filtered

    with st.sidebar.form("input_form"):
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
        job = st.selectbox("Job", options=job_options, index=job_options.index('admin.'))
        marital = st.selectbox("Marital Status", options=marital_options, index=marital_options.index('married'))
        education = st.selectbox("Education Level", options=education_options,
                                 index=education_options.index('university.degree'))

        st.subheader("Financial Information")
        default = st.selectbox("Has Credit in Default?", options=default_options, index=default_options.index('no'))
        housing = st.selectbox("Has Housing Loan?", options=housing_options, index=housing_options.index('yes'))
        loan = st.selectbox("Has Personal Loan?", options=loan_options, index=loan_options.index('no'))

        st.subheader("Last Contact Information")
        contact = st.selectbox("Contact Communication Type", options=contact_options,
                               index=contact_options.index('cellular'))
        # month and day_of_week are removed as per v2.2+ pipeline

        st.subheader("Campaign Information")
        campaign = st.number_input("Number of Contacts (This Campaign)", min_value=1, max_value=60, value=1, step=1)
        pdays = st.number_input("Days Since Last Contact (999 if never)", min_value=0, max_value=999, value=999, step=1)
        previous = st.number_input("Number of Contacts (Prior to This Campaign)", min_value=0, max_value=10, value=0,
                                   step=1)
        poutcome = st.selectbox("Outcome of Previous Campaign", options=poutcome_options,
                                index=poutcome_options.index('nonexistent'))

        st.subheader("Socioeconomic Context")
        emp_var_rate = st.number_input("Employment Variation Rate", value=1.1, format="%.1f")
        cons_price_idx = st.number_input("Consumer Price Index", value=93.994, format="%.3f")
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-36.4, format="%.1f")
        euribor3m = st.number_input("Euribor 3 Month Rate", value=4.857, format="%.3f")
        nr_employed = st.number_input("Number of Employees", value=5191.0, format="%.1f")

        submit_button = st.form_submit_button(label="Predict Subscription")

    if submit_button:
        input_data_raw = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'housing': housing, 'loan': loan, 'contact': contact,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
            'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
        }
        input_df_raw = pd.DataFrame([input_data_raw])
        input_df_engineered = engineer_features_for_app(input_df_raw)

        # Define the exact order of columns the *loaded preprocessor* expects.
        # This order is derived from the v2.4 training script:
        # numerical_features + categorical_features_to_impute + categorical_features_direct_ohe
        # The names here are *before* any "Other_" consolidation by the training script's rare category handling,
        # as the app's engineer_features doesn't do that step.

        # From v2.4 training script (section 4.2, before ColumnTransformer fitting):
        # These are the *original* or *app-engineered* column names.
        expected_numerical = sorted(
            ['age', 'campaign', 'cons.conf.idx', 'cons.price.idx', 'emp.var.rate', 'euribor3m', 'euribor_emp_rate_diff',
             'not_previously_contacted', 'nr.employed', 'pdays', 'previous'])
        expected_cat_impute = sorted(
            ['education', 'housing', 'job', 'loan', 'marital'])  # Original names designated for imputation
        expected_cat_direct_ohe = sorted(
            ['age_group', 'campaign_binned', 'contact', 'default', 'job_education', 'poutcome',
             'poutcome_x_job'])  # Original/App-Engineered names for direct OHE

        final_expected_cols_for_pipeline = expected_numerical + expected_cat_impute + expected_cat_direct_ohe
        final_expected_cols_for_pipeline = list(
            OrderedDict.fromkeys(final_expected_cols_for_pipeline))  # Ensure unique and ordered

        # Ensure all expected columns are present and in the correct order
        missing_cols = [col for col in final_expected_cols_for_pipeline if col not in input_df_engineered.columns]
        if missing_cols:
            st.error(
                f"Feature engineering error: The following columns are missing before sending to pipeline: {missing_cols}")
            st.error(f"Engineered columns available: {input_df_engineered.columns.tolist()}")
            return

        try:
            input_df_for_pipeline = input_df_engineered[final_expected_cols_for_pipeline]
        except KeyError as e:
            st.error(
                f"Column ordering/selection error before pipeline: {e}. Expected columns: {final_expected_cols_for_pipeline}")
            return

        st.subheader("Prediction Results")
        try:
            pred_proba = pipeline.predict_proba(input_df_for_pipeline)[:, 1]
            prediction_default_threshold = pipeline.predict(input_df_for_pipeline)

            st.metric(label="Probability of Subscription (Yes)", value=f"{pred_proba[0]:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Prediction (Default 0.5 Threshold)")
                if prediction_default_threshold[0] == 1:
                    st.success(" LIKELY to subscribe")
                else:
                    st.error(" UNLIKELY to subscribe")

            with col2:
                st.markdown(f"##### Prediction (Optimal Threshold: {OPTIMAL_THRESHOLD:.2f})")
                prediction_optimal_threshold = (pred_proba >= OPTIMAL_THRESHOLD).astype(int)
                if prediction_optimal_threshold[0] == 1:
                    st.success(" LIKELY to subscribe")
                else:
                    st.error(" UNLIKELY to subscribe")

            st.markdown(f"""
            **Note on Thresholds:**
            - The "Optimal Threshold" of `{OPTIMAL_THRESHOLD:.2f}` was (or should be) determined during model training to balance metrics like Precision and Recall for the 'yes' class, often by maximizing the F1-score. This can be more insightful for imbalanced datasets than the default 0.5 threshold.
            """)
            # st.dataframe(input_df_for_pipeline.T.rename(columns={0:"Input to Model (Post App FE)"}))


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure input data is valid and the model is loaded correctly.")
            st.error(f"Data sent to pipeline (first row): \n{input_df_for_pipeline.head(1).to_dict()}")
    else:
        st.info("Please fill in the customer details in the sidebar and click 'Predict Subscription'.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Based on ML pipeline v2.4")


if __name__ == "__main__":
    main()
