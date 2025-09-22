

# Name : Rameen khan
# semester : 5th 




# loan_default_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import base64

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Set page configuration
st.set_page_config(
    page_title="Loan Defaulter Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS with improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .feature-importance-plot {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .positive {
        color: #d62728;
        font-weight: bold;
    }
    .negative {
        color: #2ca02c;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    
    /* Improved dropdown and selectbox styling */
    .stSelectbox > div > div {
        background-color: #e6f3ff !important;
        border-radius: 5px;
        border: 1px solid #1f77b4;
    }
    
    .stSlider > div > div {
        background-color: #e6f3ff !important;
        border-radius: 5px;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Card-like containers */
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #1668a3;
        color: white;
    }
    
    /* Success and error messages */
    .stSuccess {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
    }
    
    .stError {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 15px;
        border-radius: 5px;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class LoanDefaultPredictor:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_samples=1000):
        # Generate synthetic data for the loan default problem
        np.random.seed(42)
        
        # Create synthetic features
        age = np.random.normal(35, 10, n_samples).astype(int)
        income = np.random.normal(50000, 20000, n_samples).astype(int)
        loan_amount = np.random.normal(10000, 5000, n_samples).astype(int)
        credit_score = np.random.normal(650, 100, n_samples).astype(int)
        employment_length = np.random.exponential(5, n_samples).astype(int)
        debt_to_income = np.random.uniform(0.1, 0.8, n_samples)
        existing_loans = np.random.poisson(1.5, n_samples).astype(int)
        
        # Create categorical features
        home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, p=[0.3, 0.4, 0.3])
        loan_purpose = np.random.choice(['CAR', 'HOME', 'EDUCATION', 'MEDICAL', 'PERSONAL'], n_samples)
        loan_grade = np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples, p=[0.2, 0.3, 0.25, 0.15, 0.1])
        employment_status = np.random.choice(['EMPLOYED', 'SELF-EMPLOYED', 'UNEMPLOYED'], n_samples, p=[0.7, 0.2, 0.1])
        
        # Calculate loan-to-income ratio
        loan_to_income = loan_amount / np.maximum(income, 1)
        
        # Generate target variable with some logic
        default_proba = 1 / (1 + np.exp(-(
            -2 + 
            0.05 * (age - 35) / 10 + 
            0.1 * (credit_score - 650) / 100 -
            0.2 * (income - 50000) / 20000 +
            0.3 * debt_to_income * 10 +
            0.5 * (loan_to_income > 0.4).astype(int) +
            0.4 * existing_loans +
            (home_ownership == 'RENT') * 0.3 +
            (employment_status == 'UNEMPLOYED') * 0.7 +
            (loan_grade == 'D') * 0.5 + (loan_grade == 'E') * 0.8
        )))
        
        loan_default = (default_proba > np.random.uniform(0, 1, n_samples)).astype(int)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'age': age,
            'income': income,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'employment_length': employment_length,
            'debt_to_income': debt_to_income,
            'existing_loans': existing_loans,
            'loan_to_income': loan_to_income,
            'home_ownership': home_ownership,
            'loan_purpose': loan_purpose,
            'loan_grade': loan_grade,
            'employment_status': employment_status,
            'loan_default': loan_default
        })
        
        # Ensure no negative values
        self.data['age'] = self.data['age'].clip(18, 100)
        self.data['income'] = self.data['income'].clip(10000, 200000)
        self.data['loan_amount'] = self.data['loan_amount'].clip(1000, 50000)
        self.data['credit_score'] = self.data['credit_score'].clip(300, 850)
        self.data['employment_length'] = self.data['employment_length'].clip(0, 40)
        self.data['existing_loans'] = self.data['existing_loans'].clip(0, 10)
        
        return self.data
    
    def preprocess_data(self, data):
        # Separate features and target
        self.X = data.drop('loan_default', axis=1)
        self.y = data['loan_default']
        
        # Identify categorical and numerical columns
        categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Bundle preprocessing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Store feature names for later use
        # Fit the preprocessor to get feature names
        self.preprocessor.fit(self.X)
        self.feature_names = numerical_cols + \
            list(self.preprocessor.named_transformers_['cat']\
                 .named_steps['onehot']\
                 .get_feature_names_out(categorical_cols))
        
        return self.X, self.y, self.preprocessor, self.feature_names
    
    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_name='random_forest', use_smote=False):
        # Preprocess the data
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Handle class imbalance if requested
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train_processed, self.y_train = smote.fit_resample(X_train_processed, self.y_train)
        
        # Initialize the model
        if model_name == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        elif model_name == 'logistic_regression':
            model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'svm':
            model = SVC(random_state=42, class_weight='balanced', probability=True)
        else:
            raise ValueError("Unsupported model type")
        
        # Train the model
        model.fit(X_train_processed, self.y_train)
        self.model = model
        
        return model
    
    def evaluate_model(self):
        # Preprocess the test data
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        y_pred_proba = self.model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def predict_new(self, input_data):
        # Check if model is trained
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        # Preprocess the input data
        input_processed = self.preprocessor.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(input_processed)
        prediction_proba = self.model.predict_proba(input_processed)
        
        return prediction, prediction_proba

def main():
    st.title("💰 Loan Defaulter Prediction System")
    st.markdown("""
    This application uses machine learning to predict the likelihood of a loan applicant defaulting on their loan.
    """)
    
    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None
    if 'predictor' not in st.session_state:
        st.session_state.predictor = LoanDefaultPredictor()
    
    # Create sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", 
                                   ["Data Overview", "Exploratory Data Analysis", 
                                    "Model Training", "Make Predictions", "Model Evaluation"])
    
    # Use the predictor from session state
    predictor = st.session_state.predictor
    
    # Load data
    if st.session_state.data is None:
        with st.spinner("Generating synthetic loan data..."):
            st.session_state.data = predictor.generate_synthetic_data(1000)
    
    # Preprocess data (only if not already done)
    if predictor.preprocessor is None:
        X, y, preprocessor, feature_names = predictor.preprocess_data(st.session_state.data)
        predictor.X_train, predictor.X_test, predictor.y_train, predictor.y_test = predictor.split_data()
        predictor.preprocessor = preprocessor
        predictor.feature_names = feature_names
    
    # Data Overview Section
    if app_mode == "Data Overview":
        st.header("Data Overview")
        
        st.subheader("Sample Data")
        st.dataframe(st.session_state.data.head())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Information")
            st.text(f"Shape: {st.session_state.data.shape}")
            st.text(f"Number of features: {len(st.session_state.data.columns)}")
            st.text(f"Number of records: {len(st.session_state.data)}")
            
            # Data types
            st.subheader("Data Types")
            dtypes_df = pd.DataFrame(st.session_state.data.dtypes, columns=['Data Type'])
            st.dataframe(dtypes_df)
        
        with col2:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame(st.session_state.data.isnull().sum(), columns=['Missing Values'])
            st.dataframe(missing_df)
            
            st.subheader("Class Distribution")
            class_counts = st.session_state.data['loan_default'].value_counts()
            fig = px.pie(values=class_counts.values, names=['Non-Default', 'Default'], 
                         title='Loan Default Distribution')
            st.plotly_chart(fig)
    
    # Exploratory Data Analysis Section
    elif app_mode == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = st.session_state.data.select_dtypes(include=[np.number])
        corr = numeric_data.corr()
        
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                        title="Correlation Between Numerical Features")
        st.plotly_chart(fig)
        
        # Distribution of numerical features
        st.subheader("Distribution of Numerical Features")
        num_cols = numeric_data.columns.tolist()
        selected_num_col = st.selectbox("Select a numerical feature to visualize", num_cols)
        
        fig = px.histogram(st.session_state.data, x=selected_num_col, color='loan_default',
                           marginal="box", title=f"Distribution of {selected_num_col} by Loan Default")
        st.plotly_chart(fig)
        
        # Categorical features analysis
        st.subheader("Categorical Features Analysis")
        cat_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
        
        if cat_cols:
            selected_cat_col = st.selectbox("Select a categorical feature to visualize", cat_cols)
            
            fig = px.histogram(st.session_state.data, x=selected_cat_col, color='loan_default',
                               title=f"Distribution of {selected_cat_col} by Loan Default")
            st.plotly_chart(fig)
        
        # Relationship between features
        st.subheader("Relationship Between Features")
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis", num_cols, index=0)
        with col2:
            y_axis = st.selectbox("Y-axis", num_cols, index=1)
        
        fig = px.scatter(st.session_state.data, x=x_axis, y=y_axis, color='loan_default',
                         title=f"Relationship between {x_axis} and {y_axis}")
        st.plotly_chart(fig)
    
    # Model Training Section
    elif app_mode == "Model Training":
        st.header("Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model",
                ["random_forest", "logistic_regression", "gradient_boosting", "svm"],
                key="model_type_select"
            )
        
        with col2:
            use_smote = st.checkbox("Use SMOTE for handling class imbalance", key="smote_checkbox")
        
        if st.button("Train Model", key="train_button"):
            with st.spinner("Training model..."):
                try:
                    model = predictor.train_model(model_type, use_smote)
                    st.session_state.model = model
                    evaluation = predictor.evaluate_model()
                    st.session_state.evaluation = evaluation
                    st.session_state.trained = True
                    st.session_state.predictor = predictor  # Update predictor in session state
                    
                    st.success("Model trained successfully!")
                    
                    # Display evaluation metrics
                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{evaluation['accuracy']:.4f}")
                    with col2:
                        st.metric("Precision", f"{evaluation['precision']:.4f}")
                    with col3:
                        st.metric("Recall", f"{evaluation['recall']:.4f}")
                    with col4:
                        st.metric("F1 Score", f"{evaluation['f1']:.4f}")
                    
                    st.metric("ROC AUC Score", f"{evaluation['roc_auc']:.4f}")
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_predictions(
                        predictor.y_test, evaluation['y_pred'], 
                        display_labels=['Non-Default', 'Default'], 
                        cmap='Blues', ax=ax
                    )
                    st.pyplot(fig)
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        st.subheader("Feature Importance")
                        feature_importance = pd.DataFrame({
                            'feature': predictor.feature_names,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        fig = px.bar(feature_importance.head(10), x='importance', y='feature', 
                                     title='Top 10 Important Features')
                        st.plotly_chart(fig)
                        
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
    
    # Make Predictions Section
    elif app_mode == "Make Predictions":
        st.header("Make Predictions on New Data")
        
        if not st.session_state.trained or st.session_state.model is None:
            st.warning("⚠️ Please train a model first in the 'Model Training' section.")
        else:
            st.subheader("Enter Applicant Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.slider("Age", 18, 100, 35, key="age_slider")
                income = st.slider("Annual Income ($)", 10000, 200000, 50000, key="income_slider")
                loan_amount = st.slider("Loan Amount ($)", 1000, 50000, 10000, key="loan_amount_slider")
                credit_score = st.slider("Credit Score", 300, 850, 650, key="credit_score_slider")
            
            with col2:
                employment_length = st.slider("Employment Length (years)", 0, 40, 5, key="employment_slider")
                debt_to_income = st.slider("Debt-to-Income Ratio", 0.1, 0.8, 0.4, key="dti_slider")
                existing_loans = st.slider("Existing Loans", 0, 10, 1, key="existing_loans_slider")
                home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"], key="home_ownership_select")
                loan_purpose = st.selectbox("Loan Purpose", ["CAR", "HOME", "EDUCATION", "MEDICAL", "PERSONAL"], key="loan_purpose_select")
                loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E"], key="loan_grade_select")
                employment_status = st.selectbox("Employment Status", ["EMPLOYED", "SELF-EMPLOYED", "UNEMPLOYED"], key="employment_status_select")
            
            # Calculate loan-to-income ratio
            loan_to_income = loan_amount / income if income > 0 else 0
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'income': [income],
                'loan_amount': [loan_amount],
                'credit_score': [credit_score],
                'employment_length': [employment_length],
                'debt_to_income': [debt_to_income],
                'existing_loans': [existing_loans],
                'loan_to_income': [loan_to_income],
                'home_ownership': [home_ownership],
                'loan_purpose': [loan_purpose],
                'loan_grade': [loan_grade],
                'employment_status': [employment_status]
            })
            
            if st.button("Predict", key="predict_button"):
                try:
                    prediction, prediction_proba = predictor.predict_new(input_data)
                    
                    st.subheader("Prediction Result")
                    
                    if prediction[0] == 1:
                        st.error(f"⚠️ High Risk: This applicant is likely to default (Probability: {prediction_proba[0][1]:.2%})")
                    else:
                        st.success(f"✅ Low Risk: This applicant is not likely to default (Probability: {prediction_proba[0][1]:.2%})")
                    
                    # Show probability breakdown
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Probability of Non-Default", f"{prediction_proba[0][0]:.2%}")
                    with col2:
                        st.metric("Probability of Default", f"{prediction_proba[0][1]:.2%}")
                    
                    # Show feature contributions (simplified)
                    st.subheader("Key Factors Influencing Prediction")
                    
                    # This is a simplified version - in a real scenario, you might use SHAP values
                    factors = []
                    if credit_score < 600:
                        factors.append("Low credit score")
                    if debt_to_income > 0.5:
                        factors.append("High debt-to-income ratio")
                    if loan_to_income > 0.4:
                        factors.append("High loan-to-income ratio")
                    if loan_grade in ['D', 'E']:
                        factors.append("Low loan grade")
                    if age < 25:
                        factors.append("Young applicant")
                    if employment_status == 'UNEMPLOYED':
                        factors.append("Unemployment status")
                    if existing_loans > 3:
                        factors.append("Multiple existing loans")
                    
                    if factors:
                        st.write("The following factors increase the risk of default:")
                        for factor in factors:
                            st.write(f"- {factor}")
                    else:
                        st.write("No significant risk factors identified.")
                        
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.info("Please make sure you have trained a model in the 'Model Training' section.")
    
    # Model Evaluation Section
    elif app_mode == "Model Evaluation":
        st.header("Model Evaluation")
        
        if not st.session_state.trained or st.session_state.evaluation is None:
            st.warning("⚠️ Please train a model first in the 'Model Training' section.")
        else:
            evaluation = st.session_state.evaluation
            
            # ROC Curve
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(predictor.y_test, evaluation['y_pred_proba'])
            fig = px.area(
                x=fpr, y=tpr,
                title=f'ROC Curve (AUC={evaluation["roc_auc"]:.4f})',
                labels=dict(x='False Positive Rate', y='True Positive Rate')
            )
            fig.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            st.plotly_chart(fig)
            
            # Precision-Recall Curve
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(predictor.y_test, evaluation['y_pred_proba'])
            fig = px.area(
                x=recall, y=precision,
                title=f'Precision-Recall Curve',
                labels=dict(x='Recall', y='Precision')
            )
            st.plotly_chart(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(predictor.y_test, evaluation['y_pred'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Compare different models
            st.subheader("Compare Different Models")
            if st.button("Run Model Comparison", key="compare_models_button"):
                with st.spinner("Training and comparing different models..."):
                    models = {
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
                        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                        'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
                    }
                    
                    results = []
                    
                    for name, model in models.items():
                        # Train the model
                        X_train_processed = predictor.preprocessor.transform(predictor.X_train)
                        X_test_processed = predictor.preprocessor.transform(predictor.X_test)
                        
                        model.fit(X_train_processed, predictor.y_train)
                        y_pred = model.predict(X_test_processed)
                        y_pred_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, "predict_proba") else None
                        
                        # Calculate metrics
                        accuracy = accuracy_score(predictor.y_test, y_pred)
                        precision = precision_score(predictor.y_test, y_pred)
                        recall = recall_score(predictor.y_test, y_pred)
                        f1 = f1_score(predictor.y_test, y_pred)
                        roc_auc = roc_auc_score(predictor.y_test, y_pred_proba) if y_pred_proba is not None else None
                        
                        results.append({
                            'Model': name,
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1 Score': f1,
                            'ROC AUC': roc_auc
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Display results
                    st.dataframe(results_df)
                    
                    # Plot comparison
                    fig = go.Figure()
                    
                    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    for metric in metrics:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=results_df['Model'],
                            y=results_df[metric],
                            text=results_df[metric].round(3),
                            textposition='auto'
                        ))
                    
                    fig.update_layout(
                        title='Model Comparison',
                        barmode='group',
                        yaxis=dict(title='Score', range=[0, 1])
                    )
                    
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main()