import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from collections import OrderedDict  # For preserving feature order
import joblib  # For saving the model

# For handling imbalanced data with SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE

# Set a random seed for reproducibility
RANDOM_STATE = 42
# Suppress warnings for cleaner output (optional)
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- 0. Load Data ---
print("--- 0. Loading Data ---")
try:
    df = pd.read_csv('bank-additional.csv', sep=';')
    print("Dataset loaded successfully.")
    print(f"Shape of the dataset: {df.shape}")
    df_processed = df.copy()
except FileNotFoundError:
    print("Error: 'bank-additional.csv' not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# --- 1. Initial Data Exploration & Understanding ---
print("\n--- 1. Initial Data Exploration ---")
print("\nTarget variable 'y' distribution (Original):")
print(df_processed['y'].value_counts(normalize=True))

# --- 2. Data Cleaning ---
print("\n--- 2. Data Cleaning ---")
duplicate_rows = df_processed.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicate_rows}")
if duplicate_rows > 0:
    df_processed.drop_duplicates(inplace=True)
    print(f"Dropped duplicate rows. New shape: {df_processed.shape}")

# --- 3. Feature Engineering & Selection ---
print("\n--- 3. Feature Engineering & Selection ---")

# Target variable encoding
df_processed['y_target'] = df_processed['y'].map({'no': 0, 'yes': 1})
df_processed.drop('y', axis=1, inplace=True)
print("\nTarget variable 'y' encoded to 'y_target' (0/1).")

if 'duration' in df_processed.columns:
    print("\nDropping 'duration' column.")
    df_processed.drop('duration', axis=1, inplace=True)

columns_to_drop_explicitly = ['month', 'day_of_week']
for col_drop in columns_to_drop_explicitly:
    if col_drop in df_processed.columns:
        df_processed.drop(col_drop, axis=1, inplace=True)
        print(f"Dropped '{col_drop}' column.")

if 'pdays' in df_processed.columns:
    df_processed['not_previously_contacted'] = (df_processed['pdays'] == 999).astype(int)
    df_processed.loc[df_processed['pdays'] == 999, 'pdays'] = -1
    print("\nEngineered 'not_previously_contacted'; Transformed 'pdays'.")

age_bins = [0, 30, 40, 50, 60, 100]
age_labels = ['Young', 'Adult', 'Mid-Adult', 'Senior', 'Elderly']
if 'age' in df_processed.columns:
    df_processed['age_group'] = pd.cut(df_processed['age'], bins=age_bins, labels=age_labels, right=False)
    print("Engineered 'age_group'.")

campaign_bins = [0, 1, 2, 3, 5, 10, 100]
campaign_labels = ['1_contact', '2_contacts', '3_contacts', '4-5_contacts', '6-10_contacts', '>10_contacts']
if 'campaign' in df_processed.columns:
    df_processed['campaign_binned'] = pd.cut(df_processed['campaign'], bins=campaign_bins, labels=campaign_labels,
                                             right=True, include_lowest=True)
    df_processed['campaign_binned'] = df_processed['campaign_binned'].astype(object).fillna('Unknown_Campaign_Bin')
    print("Engineered 'campaign_binned'.")

if 'job' in df_processed.columns and 'education' in df_processed.columns:
    df_processed['job_education'] = df_processed['job'].astype(str) + "_" + df_processed['education'].astype(str)
    print("Engineered 'job_education'.")

if 'euribor3m' in df_processed.columns and 'emp.var.rate' in df_processed.columns:
    df_processed['euribor_emp_rate_diff'] = df_processed['euribor3m'] - df_processed['emp.var.rate']
    print("Engineered 'euribor_emp_rate_diff'.")

if 'poutcome' in df_processed.columns and 'job' in df_processed.columns:
    df_processed['poutcome_x_job'] = df_processed['poutcome'].astype(str) + "_" + df_processed['job'].astype(str)
    print("Engineered 'poutcome_x_job'.")

# --- 4. Data Splitting & Initial Feature Type Identification ---
print("\n--- 4. Data Splitting & Initial Feature Type Identification ---")
X_initial = df_processed.drop('y_target', axis=1)
y_initial = df_processed['y_target']

initial_categorical_cols = X_initial.select_dtypes(include=['object', 'category']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_initial, y_initial, test_size=0.25, random_state=RANDOM_STATE,
                                                    stratify=y_initial)
print(f"\nData split: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# --- 4.1 Rare Category Consolidation ---
print("\n--- 4.1 Rare Category Consolidation ---")
rare_threshold_pct = 0.01  # Example: categories < 1% of training data are rare
min_count_threshold = 10  # Example: categories with < 10 occurrences are rare

# Make copies to avoid SettingWithCopyWarning if X_train/X_test are views
X_train = X_train.copy()
X_test = X_test.copy()

for col in initial_categorical_cols:
    if col in X_train.columns:
        value_counts_train = X_train[col].value_counts()
        n_samples_train = len(X_train)

        rare_categories = value_counts_train[
            (value_counts_train < min_count_threshold) | (value_counts_train / n_samples_train < rare_threshold_pct)
            ].index.tolist()

        if rare_categories:
            print(f"Consolidating {len(rare_categories)} rare categories in '{col}' into 'Other_{col}'.")
            other_category_name = f"Other_{col}"
            X_train.loc[:, col] = X_train[col].replace(rare_categories, other_category_name)
            X_test.loc[:, col] = X_test[col].replace(rare_categories, other_category_name)
        else:
            print(f"No rare categories to consolidate in '{col}'.")

# --- 4.2 Final Feature Type Identification (Post Rare Category Consolidation) ---
print("\n--- 4.2 Final Feature Type Identification ---")
numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
if 'not_previously_contacted' in X_train.columns and 'not_previously_contacted' not in numerical_features: numerical_features.append(
    'not_previously_contacted')
if 'euribor_emp_rate_diff' in X_train.columns and 'euribor_emp_rate_diff' not in numerical_features: numerical_features.append(
    'euribor_emp_rate_diff')
numerical_features = sorted(list(set(numerical_features)))

all_object_category_cols_post_rare = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_features_to_impute = []
categorical_features_direct_ohe = []
designated_for_imputation = ['job', 'marital', 'education', 'housing', 'loan']

for col in all_object_category_cols_post_rare:
    original_col_name_for_check = col.split("Other_")[-1]
    if original_col_name_for_check in designated_for_imputation:
        categorical_features_to_impute.append(col)
    else:
        categorical_features_direct_ohe.append(col)
categorical_features_to_impute = sorted(list(set(categorical_features_to_impute)))
categorical_features_direct_ohe = sorted(list(set(categorical_features_direct_ohe)))

print(f"\nNumerical features for RobustScaling: {numerical_features}")
print(f"Categorical features for Mode Imputation (add_indicator=True) then OHE: {categorical_features_to_impute}")
print(f"Categorical features for Direct OHE: {categorical_features_direct_ohe}")

# --- 4.3 Preprocessor Setup ---
numerical_transformer = RobustScaler()
categorical_impute_pipeline = SklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),  # add_indicator=True
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
])
categorical_direct_ohe_transformer = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)

transformers_list_ct = []
if numerical_features:
    transformers_list_ct.append(('num', numerical_transformer, numerical_features))
if categorical_features_to_impute:
    transformers_list_ct.append(('cat_impute_ohe', categorical_impute_pipeline, categorical_features_to_impute))
if categorical_features_direct_ohe:
    transformers_list_ct.append(('cat_direct_ohe', categorical_direct_ohe_transformer, categorical_features_direct_ohe))

if not transformers_list_ct:
    print("Error: No features defined for preprocessing. Exiting.")
    exit()
preprocessor = ColumnTransformer(transformers=transformers_list_ct, remainder='passthrough',
                                 verbose_feature_names_out=False)  # verbose_feature_names_out=False for cleaner names

expected_input_features_to_preprocessor_ordered = numerical_features + categorical_features_to_impute + categorical_features_direct_ohe
expected_input_features_to_preprocessor_ordered = list(
    OrderedDict.fromkeys(expected_input_features_to_preprocessor_ordered))
X_train = X_train[expected_input_features_to_preprocessor_ordered]
X_test = X_test[expected_input_features_to_preprocessor_ordered]
print(f"Shape of X_train after column selection for preprocessor: {X_train.shape}")

# --- 5. Model Selection, Tuning & Imbalance Handling ---
print("\n--- 5. Model Selection, Tuning & Imbalance Handling ---")
count_neg_train = y_train.value_counts().get(0, 0)
count_pos_train = y_train.value_counts().get(1, 0)
scale_pos_weight_val = 1
if count_pos_train > 0:
    scale_pos_weight_val = count_neg_train / count_pos_train
    print(f"Calculated scale_pos_weight for XGB/LGBM: {scale_pos_weight_val:.2f}")
else:
    print("Warning: No positive samples in training data for scale_pos_weight.")

models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss',
                                 scale_pos_weight=scale_pos_weight_val),
    "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, scale_pos_weight=scale_pos_weight_val)
}
param_grids = {  # Reduced grids for quicker demonstration; expand for thorough search
    "Logistic Regression": {'classifier__C': [0.001, 0.01, 0.1, 1]},
    "Random Forest": {
        'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 20, None],
        'classifier__min_samples_split': [2, 5], 'classifier__min_samples_leaf': [1, 2]
    },
    "Gradient Boosting": {
        'classifier__n_estimators': [100, 200], 'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    },
    "XGBoost": {
        'classifier__n_estimators': [100, 150], 'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5], 'classifier__gamma': [0, 0.1]
    },
    "LightGBM": {
        'classifier__n_estimators': [100, 150], 'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [-1, 5, 10], 'classifier__num_leaves': [20, 31, 50]
    }
}
cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
best_models_info = {}

min_samples_minority_class_train = y_train.value_counts().min()
smote_k_neighbors = max(1, min(5, min_samples_minority_class_train - 1))
use_smote_for_non_boosting_models = False
if min_samples_minority_class_train > smote_k_neighbors:
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_neighbors)
    use_smote_for_non_boosting_models = True
    print(f"SMOTE will be used for LR, RF, GB with k_neighbors={smote_k_neighbors}.")
else:
    print(
        f"Warning: Minority class in training set ({min_samples_minority_class_train} samples) is too small for SMOTE. SMOTE will be skipped for LR, RF, GB.")

for model_name, model_instance in models.items():
    print(f"\n--- Training and tuning {model_name} ---")
    n_jobs_cv = -1
    if model_name in ["XGBoost", "LightGBM"]:
        n_jobs_cv = 1
        current_pipeline = SklearnPipeline(steps=[('preprocessor', preprocessor), ('classifier', model_instance)])
    else:
        if use_smote_for_non_boosting_models:
            current_pipeline = ImblearnPipeline(
                steps=[('preprocessor', preprocessor), ('smote', smote), ('classifier', model_instance)])
        else:
            current_pipeline = SklearnPipeline(steps=[('preprocessor', preprocessor), ('classifier', model_instance)])
            if model_name == "Gradient Boosting":
                print(f"   Note: SMOTE skipped for {model_name}.")
    grid_search = GridSearchCV(current_pipeline, param_grids[model_name], cv=cv_stratified,
                               scoring='average_precision', n_jobs=n_jobs_cv, verbose=1, refit='average_precision')
    try:
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_pr_auc = grid_search.best_score_
        y_pred_test = best_estimator.predict(X_test)
        y_proba_test = best_estimator.predict_proba(X_test)[:, 1]
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        recall_test = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        f1_test = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)
        roc_auc_test = roc_auc_score(y_test, y_proba_test)
        pr_auc_test = average_precision_score(y_test, y_proba_test)
        best_models_info[model_name] = {
            "Best Estimator": best_estimator, "Best Params": best_params,
            "Best CV PR AUC": best_cv_pr_auc, "Test Accuracy": accuracy_test,
            "Test Precision (Class 1)": precision_test, "Test Recall (Class 1)": recall_test,
            "Test F1-score (Class 1)": f1_test, "Test ROC AUC": roc_auc_test,
            "Test PR AUC": pr_auc_test
        }
        print(f"\nResults for {model_name}:")
        print(f"  Best PR AUC (Cross-Validation): {best_cv_pr_auc:.4f}")
        print(f"  Best Hyperparameters: {best_params}")
        print(
            f"  Test Set Performance: Acc={accuracy_test:.4f}, P={precision_test:.4f}, R={recall_test:.4f}, F1={f1_test:.4f}, ROC_AUC={roc_auc_test:.4f}, PR_AUC={pr_auc_test:.4f}")
    except Exception as e:
        print(f"An error occurred with {model_name}: {e}")

# --- 6. Select the Best Model & Final Pipeline ---
print("\n--- 6. Model Comparison & Final Selection (based on Test PR AUC) ---")
if not best_models_info:
    print("No models were successfully trained. Exiting.")
    exit()
results_list = []
for model_name_iter, data_iter in best_models_info.items():
    results_list.append({
        "Model": model_name_iter, "Best CV PR AUC": data_iter.get("Best CV PR AUC"),
        "Test PR AUC": data_iter.get("Test PR AUC"), "Test ROC AUC": data_iter.get("Test ROC AUC"),
        "Test Recall (Class 1)": data_iter.get("Test Recall (Class 1)"),
        "Test F1-score (Class 1)": data_iter.get("Test F1-score (Class 1)"),
        "Test Precision (Class 1)": data_iter.get("Test Precision (Class 1)"),
        "Test Accuracy": data_iter.get("Test Accuracy"),
    })
results_df = pd.DataFrame(results_list).sort_values(by="Test PR AUC", ascending=False)
print("\nModel Performance Summary (Sorted by Test PR AUC):")
print(results_df)

if results_df.empty or results_df["Test PR AUC"].isnull().all():
    print("No model results with valid Test PR AUC. Exiting.")
    exit()
best_model_name_overall = results_df.iloc[0]["Model"]
final_best_model_pipeline = best_models_info[best_model_name_overall]["Best Estimator"]
print(f"\nBest performing model based on Test PR AUC: {best_model_name_overall}")
print(f"Its parameters: {best_models_info[best_model_name_overall]['Best Params']}")
print(f"\nFinal Pipeline structure for {best_model_name_overall}:\n{final_best_model_pipeline}")

# Save the best model
try:
    model_filename = f'{best_model_name_overall.replace(" ", "_")}_pipeline_v2_4.pkl'  # Updated version in filename
    joblib.dump(final_best_model_pipeline, model_filename)
    print(f"\nBest model pipeline saved to '{model_filename}'")
except Exception as e:
    print(f"Error saving the model: {e}")

# --- 7. Feature Importances (Simplified) ---
print(f"\n--- 7. Feature Importances for {best_model_name_overall} (if available) ---")
print("Note: Accurate feature name retrieval from complex preprocessors is challenging.")
print("The following importances/coefficients are against transformed features.")
try:
    # Get the preprocessor and classifier steps from the best pipeline
    fitted_preprocessor = final_best_model_pipeline.named_steps.get('preprocessor')
    best_classifier_step = final_best_model_pipeline.named_steps.get('classifier')

    if fitted_preprocessor and best_classifier_step:
        # Attempt to get feature names from the fitted preprocessor
        try:
            # This method is more robust for scikit-learn >= 0.24
            # For ColumnTransformer, get_feature_names_out() is preferred
            if hasattr(fitted_preprocessor, 'get_feature_names_out'):
                retrieved_feature_names = fitted_preprocessor.get_feature_names_out()
            else:  # Fallback for older versions or if it's a simple pipeline step
                # This part is highly dependent on the structure and might need manual adjustment
                # For a ColumnTransformer, this fallback is tricky.
                # We'll try a simplified approach assuming direct access if not ColumnTransformer
                if hasattr(fitted_preprocessor, 'transformers_'):
                    retrieved_feature_names = []
                    for name, trans, L_cols in fitted_preprocessor.transformers_:
                        if trans == 'drop' or trans == 'passthrough' and not L_cols: continue
                        if hasattr(trans, 'get_feature_names_out'):
                            if isinstance(trans, OneHotEncoder):  # OHE in a sub-pipeline
                                sub_pipeline_ohe = trans
                                # Need to know which original columns were passed to this OHE
                                # This is where it gets complex if OHE is nested.
                                # Let's assume L_cols are the original names for this OHE step.
                                retrieved_feature_names.extend(sub_pipeline_ohe.get_feature_names_out(L_cols))
                            else:  # Other transformers in sub-pipelines
                                retrieved_feature_names.extend(trans.get_feature_names_out())

                        elif isinstance(trans, SklearnPipeline):  # If a sub-pipeline (like categorical_impute_pipeline)
                            # Try to get names from the last step of the sub-pipeline (usually OHE)
                            last_step_in_sub_pipeline = trans.steps[-1][1]
                            if hasattr(last_step_in_sub_pipeline, 'get_feature_names_out'):
                                # Need the input features to this sub-pipeline (L_cols)
                                # And the input features to the OHE step within the sub-pipeline
                                # This requires knowing the structure precisely.
                                # For categorical_impute_pipeline, L_cols are input to SimpleImputer.
                                # The SimpleImputer output (with potential indicator) goes to OHE.
                                # This is hard to generalize perfectly here.
                                # A simpler approach: if the last step is OHE, use L_cols as input_features.
                                if isinstance(last_step_in_sub_pipeline, OneHotEncoder):
                                    retrieved_feature_names.extend(
                                        last_step_in_sub_pipeline.get_feature_names_out(L_cols))
                                else:  # Fallback for other last steps
                                    retrieved_feature_names.extend(L_cols)  # Simplification
                            else:
                                retrieved_feature_names.extend(L_cols)  # Simplification
                        else:  # e.g. RobustScaler
                            retrieved_feature_names.extend(L_cols)
                else:  # Not a ColumnTransformer, maybe a single transformer
                    if hasattr(fitted_preprocessor, 'get_feature_names_out'):
                        retrieved_feature_names = fitted_preprocessor.get_feature_names_out()
                    else:
                        retrieved_feature_names = ["feature_" + str(i) for i in range(
                            best_classifier_step.feature_importances_.shape[0] if hasattr(best_classifier_step,
                                                                                          'feature_importances_') else
                            best_classifier_step.coef_.shape[1])]

            retrieved_feature_names = list(retrieved_feature_names)  # Ensure it's a list

        except Exception as e_fn:
            print(f"    Could not automatically retrieve all feature names: {e_fn}. Using generic names.")
            # Fallback to generic feature names if specific names can't be retrieved
            num_features_expected = 0
            if hasattr(best_classifier_step, 'feature_importances_'):
                num_features_expected = len(best_classifier_step.feature_importances_)
            elif hasattr(best_classifier_step, 'coef_'):
                num_features_expected = len(best_classifier_step.coef_[0])
            retrieved_feature_names = [f"feature_{i}" for i in range(num_features_expected)]

        if hasattr(best_classifier_step, 'feature_importances_'):
            importances = best_classifier_step.feature_importances_
            if len(retrieved_feature_names) == len(importances):
                feature_importance_df = pd.DataFrame({'feature': retrieved_feature_names, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
                print("\nTop 10 Feature Importances:")
                print(feature_importance_df.head(10))
            else:
                print(
                    f"Mismatch in feature names ({len(retrieved_feature_names)}) and importances ({len(importances)}). Displaying raw importances.")
                print(importances[:10])  # Show first 10 raw importances

        elif hasattr(best_classifier_step, 'coef_'):
            coefficients = best_classifier_step.coef_[0]
            if len(retrieved_feature_names) == len(coefficients):
                feature_coeffs_df = pd.DataFrame({'feature': retrieved_feature_names, 'coefficient': coefficients})
                feature_coeffs_df['abs_coefficient'] = np.abs(feature_coeffs_df['coefficient'])
                feature_coeffs_df = feature_coeffs_df.sort_values(by='abs_coefficient', ascending=False)
                print("\nTop 10 Feature Coefficients (Absolute Value):")
                print(feature_coeffs_df.head(10))
            else:
                print(
                    f"Mismatch in feature names ({len(retrieved_feature_names)}) and coefficients ({len(coefficients)}). Displaying raw coefficients.")
                print(coefficients[:10])  # Show first 10 raw coefficients
        else:
            print(
                f"The best model ({best_model_name_overall}) does not have standard 'feature_importances_' or 'coef_' attributes.")
    else:
        print("Could not access preprocessor or classifier step from the best pipeline to show feature importances.")
except Exception as e:
    print(f"Error retrieving feature importances: {e}")

print("\n--- End of Improved ML Pipeline Script (v2.4 - Refined Imputation & Model Saving")