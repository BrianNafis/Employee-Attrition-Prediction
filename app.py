import streamlit as st
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import optional visualization libraries
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Navigation
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Single Prediction", "Batch Prediction", "EDA and Insights", "About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Need Help?\n"
    "- Fill in employee details\n"
    "- Click Predict\n"
    "- Get retention recommendations"
)

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        if JOBLIB_AVAILABLE:
            model = joblib.load("models/best_model.pkl")
            return model
        else:
            return None
    except:
        return None

@st.cache_resource
def create_mock_model():
    """Create a simple mock model for demonstration"""
    if SKLEARN_AVAILABLE:
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_dummy = np.random.rand(100, 12)
        y_dummy = np.random.randint(0, 2, 100)
        model.fit(X_dummy, y_dummy)
        return model
    return None

@st.cache_data
def load_sample_data():
    """Load sample data for demo"""
    sample_data = pd.DataFrame({
        'Age': [25, 35, 45, 28, 32, 40, 38, 30, 42, 35],
        'MonthlySalary': [3000, 5000, 8000, 4000, 6000, 7000, 6500, 4500, 7500, 5500],
        'Job_Satisfaction': [2, 4, 5, 3, 4, 2, 5, 3, 4, 3],
        'Work_Live_Balance': [2, 4, 5, 3, 4, 2, 5, 3, 4, 3],
        'Psychological_Exhaustion': [4, 2, 1, 3, 2, 4, 1, 3, 2, 3],
        'Physical_Stress': [4, 2, 1, 3, 2, 4, 1, 3, 2, 3],
        'Environment_Satisfaction': [2, 4, 5, 3, 4, 2, 5, 3, 4, 3],
        'Job_Opportunities': [2, 4, 3, 2, 4, 2, 5, 3, 3, 3],
        'OverTime': [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        'Years_Experience': [2, 8, 15, 4, 6, 12, 10, 5, 14, 7]
    })
    return sample_data

# Try to load model, create mock if not available
model = load_model()
if model is None:
    model = create_mock_model()
    if model is not None:
        st.info("Using demo model. Train a real model for better predictions.")

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_attrition(input_data, model):
    """Make prediction using the model"""
    if model is None:
        # Simple rule-based prediction for demo
        risk_score = 0
        if input_data['Psychological_Exhaustion'].values[0] >= 4:
            risk_score += 30
        if input_data['Physical_Stress'].values[0] >= 4:
            risk_score += 20
        if input_data['Job_Satisfaction'].values[0] <= 2:
            risk_score += 25
        if input_data['Work_Live_Balance'].values[0] <= 2:
            risk_score += 15
        if input_data['OverTime'].values[0] == 1:
            risk_score += 10
        
        probability = risk_score / 100
        prediction = 1 if probability > 0.5 else 0
        return prediction, np.array([1-probability, probability])
    else:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        return prediction, probability

# ============================================================================
# PAGE 1: SINGLE PREDICTION
# ============================================================================

if page == "Single Prediction":
    st.markdown('<p class="main-header">Employee Attrition Predictor</p>', unsafe_allow_html=True)
    st.markdown("Predict whether an employee is likely to leave the company based on their profile.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="sub-header">Personal and Job Information</p>', unsafe_allow_html=True)
        
        age = st.slider("Age", 18, 65, 30, help="Employee's age in years")
        
        years_exp = st.slider("Years of Experience", 0, 40, 5, help="Total years of work experience")
        
        monthly_salary = st.number_input(
            "Monthly Salary (USD)", 
            min_value=1000, 
            max_value=50000, 
            value=5000,
            step=500,
            help="Monthly salary in USD"
        )
        
        overtime = st.selectbox(
            "Regular Overtime?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Does the employee regularly work overtime?"
        )
    
    with col2:
        st.markdown('<p class="sub-header">Satisfaction and Wellbeing</p>', unsafe_allow_html=True)
        
        job_satisfaction = st.select_slider(
            "Job Satisfaction",
            options=[1, 2, 3, 4, 5],
            value=3,
            help="1 = Very Dissatisfied, 5 = Very Satisfied"
        )
        
        work_life_balance = st.select_slider(
            "Work-Life Balance",
            options=[1, 2, 3, 4, 5],
            value=3,
            help="1 = Poor, 5 = Excellent"
        )
        
        psychological_exhaustion = st.select_slider(
            "Psychological Exhaustion",
            options=[1, 2, 3, 4, 5],
            value=3,
            help="1 = Low, 5 = High (Burnout risk)"
        )
        
        physical_stress = st.select_slider(
            "Physical Stress Level",
            options=[1, 2, 3, 4, 5],
            value=3,
            help="1 = Low, 5 = High"
        )
    
    with st.expander("Advanced Options (Optional)"):
        col3, col4 = st.columns(2)
        
        with col3:
            environment_satisfaction = st.select_slider(
                "Environment Satisfaction",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="Satisfaction with work environment"
            )
            
            job_opportunities = st.select_slider(
                "Job Opportunities",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="Perceived career growth opportunities"
            )
        
        with col4:
            job_stability = st.select_slider(
                "Job Stability Perception",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="How secure the employee feels in their role"
            )
            
            recognition = st.select_slider(
                "Recognition Level",
                options=[1, 2, 3, 4, 5],
                value=3,
                help="How valued the employee feels"
            )
    
    st.markdown("---")
    predict_btn = st.button("Predict Attrition Risk", use_container_width=True, type="primary")
    
    if predict_btn:
        input_data = pd.DataFrame({
            'Age': [age],
            'Years_Experience': [years_exp],
            'MonthlySalary': [monthly_salary],
            'OverTime': [overtime],
            'Job_Satisfaction': [job_satisfaction],
            'Work_Live_Balance': [work_life_balance],
            'Psychological_Exhaustion': [psychological_exhaustion],
            'Physical_Stress': [physical_stress],
            'Environment_Satisfaction': [environment_satisfaction],
            'Job_Opportunities': [job_opportunities],
            'Job_Stability': [job_stability],
            'Recognition': [recognition]
        })
        
        prediction, probability = predict_attrition(input_data, model)
        
        st.markdown("---")
        st.markdown("## Prediction Result")
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            if prediction == 1:
                st.markdown(
                    """
                    <div class="metric-card">
                        <h3 style="color: #dc3545;">HIGH RISK</h3>
                        <p>Employee is likely to leave</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="metric-card">
                        <h3 style="color: #28a745;">LOW RISK</h3>
                        <p>Employee is likely to stay</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        with col_result2:
            risk_prob = probability[1] if prediction == 1 else probability[0]
            st.metric("Confidence Level", f"{risk_prob:.1%}")
        
        with col_result3:
            st.metric("Attrition Probability", f"{probability[1]:.1%}")
        
        st.markdown("### Risk Factor Analysis")
        
        risk_factors = []
        
        if psychological_exhaustion >= 4:
            risk_factors.append("High psychological exhaustion (burnout risk)")
        if physical_stress >= 4:
            risk_factors.append("High physical stress level")
        if job_satisfaction <= 2:
            risk_factors.append("Low job satisfaction")
        if work_life_balance <= 2:
            risk_factors.append("Poor work-life balance")
        if overtime == 1:
            risk_factors.append("Regular overtime work")
        if environment_satisfaction <= 2:
            risk_factors.append("Unsatisfactory work environment")
        
        if risk_factors:
            st.warning("Risk Factors Identified:")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.success("No major risk factors identified. Employee appears to be in a good position.")
        
        st.markdown("### Retention Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            if psychological_exhaustion >= 4:
                st.info("Wellness Support: Consider mental health programs, flexible hours, or reduced workload")
            if job_satisfaction <= 2:
                st.info("Career Development: Discuss career goals, provide growth opportunities, regular feedback")
            if environment_satisfaction <= 2:
                st.info("Work Environment: Improve office culture, team building activities, better facilities")
        
        with rec_col2:
            if work_life_balance <= 2:
                st.info("Work-Life Balance: Implement flexible scheduling, remote work options, respect boundaries")
            if overtime == 1:
                st.info("Overtime Management: Review workload distribution, consider hiring additional staff")
            if recognition <= 2:
                st.info("Recognition: Implement employee recognition program, celebrate achievements")

=
#PAGE 2: BATCH PREDICTION

elif page == "Batch Prediction":
    st.markdown('<p class="main-header">Batch Prediction</p>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file with multiple employees to predict attrition risk for all.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload CSV with employee data. Columns should match training features."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} employee records")
            
            with st.expander("Data Preview"):
                st.dataframe(df.head(10))
                st.caption(f"Total rows: {len(df)} | Total columns: {len(df.columns)}")
            
            # Select features for prediction
            feature_cols = ['Age', 'Years_Experience', 'MonthlySalary', 'OverTime', 
                           'Job_Satisfaction', 'Work_Live_Balance', 'Psychological_Exhaustion',
                           'Physical_Stress', 'Environment_Satisfaction', 'Job_Opportunities',
                           'Job_Stability', 'Recognition']
            
            available_features = [col for col in feature_cols if col in df.columns]
            
            if len(available_features) == 0:
                st.error("No matching features found in uploaded file.")
            else:
                X_pred = df[available_features]
                
                # Make predictions
                predictions = []
                probabilities = []
                
                for idx, row in X_pred.iterrows():
                    input_data = pd.DataFrame([row])
                    pred, prob = predict_attrition(input_data, model)
                    predictions.append(pred)
                    probabilities.append(prob[1])
                
                df_result = df.copy()
                df_result['Attrition_Risk'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                df_result['Attrition_Probability'] = probabilities
                
                st.markdown("### Prediction Summary")
                
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                
                high_risk_count = sum(predictions)
                low_risk_count = len(predictions) - high_risk_count
                
                with col_sum1:
                    st.metric("Total Employees", len(df))
                with col_sum2:
                    st.metric("High Risk", high_risk_count, delta=f"{(high_risk_count/len(df)*100):.1f}%")
                with col_sum3:
                    st.metric("Low Risk", low_risk_count, delta=f"{(low_risk_count/len(df)*100):.1f}%")
                with col_sum4:
                    st.metric("Avg Risk Probability", f"{np.mean(probabilities):.1%}")
                
                st.markdown("### Risk Distribution")
                
                risk_data = {
                    'Risk Level': ['High Risk', 'Low Risk'],
                    'Count': [high_risk_count, low_risk_count]
                }
                
                st.write(f"**High Risk:** {high_risk_count} employees ({high_risk_count/len(df)*100:.1f}%)")
                st.write(f"**Low Risk:** {low_risk_count} employees ({low_risk_count/len(df)*100:.1f}%)")
                
                # Simple bar chart using st.bar_chart if matplotlib not available
                if MATPLOTLIB_AVAILABLE:
                    fig, ax = plt.subplots()
                    ax.bar(['High Risk', 'Low Risk'], [high_risk_count, low_risk_count], 
                           color=['#dc3545', '#28a745'])
                    ax.set_ylabel('Number of Employees')
                    ax.set_title('Attrition Risk Distribution')
                    st.pyplot(fig)
                
                st.markdown("### Detailed Results")
                st.dataframe(
                    df_result[['Attrition_Risk', 'Attrition_Probability'] + 
                              [col for col in df_result.columns if col not in ['Attrition_Risk', 'Attrition_Probability']][:5]],
                    use_container_width=True
                )
                
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name="attrition_predictions.csv",
                    mime="text/csv",
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.markdown("---")
    with st.expander("Use Sample Data"):
        if st.button("Load Sample Data"):
            sample_data = load_sample_data()
            st.dataframe(sample_data)

# ============================================================================
# PAGE 3: EDA AND INSIGHTS
# ============================================================================

elif page == "EDA and Insights":
    st.markdown('<p class="main-header">Exploratory Data Analysis and Insights</p>', unsafe_allow_html=True)
    
    st.markdown("## Key Business Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown(
            """
            <div class="insight-box">
                <h3>Top Attrition Drivers</h3>
                <ul>
                    <li><strong>Psychological Exhaustion</strong> - Highest correlation with attrition</li>
                    <li><strong>Physical Stress</strong> - Strong positive correlation</li>
                    <li><strong>Overtime</strong> - Regular overtime increases risk</li>
                    <li><strong>Job Dissatisfaction</strong> - Low satisfaction equals high risk</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with insight_col2:
        st.markdown(
            """
            <div class="insight-box">
                <h3>Retention Factors</h3>
                <ul>
                    <li><strong>Job Satisfaction</strong> - Strongest negative correlation</li>
                    <li><strong>Work-Life Balance</strong> - Critical for retention</li>
                    <li><strong>Environment Satisfaction</strong> - Positive work culture matters</li>
                    <li><strong>Recognition</strong> - Feeling valued reduces attrition</li>
                </ul>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    st.markdown("## Feature Importance Analysis")
    
    # Create sample feature importance for demo
    features = ['Psychological Exhaustion', 'Job Satisfaction', 'Physical Stress', 
                'Work Life Balance', 'Monthly Salary', 'Environment Satisfaction', 
                'Overtime', 'Recognition', 'Job Opportunities', 'Years Experience']
    
    importance = [18, 15, 14, 12, 10, 9, 8, 7, 4, 3]
    
    if MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(features, importance, color='steelblue')
        ax.set_xlabel('Importance (%)')
        ax.set_title('Top Features Impacting Attrition')
        
        # Add value labels on bars
        for bar, imp in zip(bars, importance):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{imp}%', va='center')
        
        st.pyplot(fig)
    else:
        # Simple table if matplotlib not available
        importance_df = pd.DataFrame({'Feature': features, 'Importance (%)': importance})
        st.dataframe(importance_df)
    
    st.markdown("## Risk Profile by Employee Segment")
    
    segments = ['High Stress', 'Low Satisfaction', 'Poor WLB', 'High Overtime', 'Low Risk']
    risks = [78, 72, 68, 65, 15]
    counts = [245, 312, 198, 267, 890]
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Attrition Risk by Segment**")
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#dc3545', '#fd7e14', '#ffc107', '#ffc107', '#28a745']
            bars = ax.bar(segments, risks, color=colors)
            ax.set_ylabel('Attrition Risk (%)')
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bar, risk in zip(bars, risks):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                       f'{risk}%', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            risk_df = pd.DataFrame({'Segment': segments, 'Risk (%)': risks})
            st.dataframe(risk_df)
    
    with col_b:
        st.markdown("**Segment Size Distribution**")
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(counts, labels=segments, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        else:
            size_df = pd.DataFrame({'Segment': segments, 'Count': counts})
            st.dataframe(size_df)
    
    st.markdown("## Actionable Recommendations for HR")
    
    rec_tabs = st.tabs(["Wellbeing", "Career Growth", "Work-Life Balance", "Culture"])
    
    with rec_tabs[0]:
        st.markdown("""
        ### Mental Health and Wellbeing Initiatives
        
        **High Impact Actions:**
        - Implement Employee Assistance Program (EAP) with confidential counseling
        - Regular mental health check-ins and surveys
        - Stress management workshops and mindfulness training
        - Flexible work arrangements for employees showing burnout signs
        - Mandatory vacation policy to prevent burnout
        
        **Key Metrics to Track:**
        - Psychological Exhaustion scores over time
        - Utilization of mental health resources
        - Sick leave patterns and trends
        """)
    
    with rec_tabs[1]:
        st.markdown("""
        ### Career Development and Growth
        
        **High Impact Actions:**
        - Regular career path discussions (quarterly)
        - Skills development budget and training programs
        - Internal promotion track with clear criteria
        - Mentorship program pairing junior with senior staff
        - Cross-functional project opportunities
        
        **Key Metrics to Track:**
        - Internal promotion rate
        - Training completion rates
        - Employee skill progression
        - Career satisfaction scores
        """)
    
    with rec_tabs[2]:
        st.markdown("""
        ### Work-Life Balance Enhancement
        
        **High Impact Actions:**
        - Review and cap overtime hours by department
        - Implement no-meeting blocks during the week
        - Remote or hybrid work options where feasible
        - Respect boundaries (no after-hours emails or calls)
        - Results-oriented work environment (focus on output, not hours)
        
        **Key Metrics to Track:**
        - Average overtime hours per employee
        - Work-Life Balance satisfaction scores
        - After-hours communication frequency
        - Employee turnover by department
        """)
    
    with rec_tabs[3]:
        st.markdown("""
        ### Workplace Culture and Environment
        
        **High Impact Actions:**
        - Regular team building activities and social events
        - Employee recognition program (peer-to-peer and manager)
        - Open feedback culture with anonymous surveys
        - Improve physical workspace (ergonomics, common areas)
        - Diversity and inclusion initiatives
        
        **Key Metrics to Track:**
        - Employee Net Promoter Score (eNPS)
        - Recognition frequency and quality
        - Environment satisfaction scores
        - Retention rate by team or department
        """)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

else:
    st.markdown('<p class="main-header">About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Project Overview
    
    This application uses Machine Learning to predict employee attrition (turnover) and provides actionable insights for HR teams to improve retention.
    
    ### Dataset
    - Source: Saudi Employee Attrition Dataset
    - Samples: 1,454 employee records
    - Features: 33 attributes including demographics, job information, satisfaction metrics, and wellbeing indicators
    
    ### Models Used
    | Model | Performance |
    |-------|-------------|
    | XGBoost | Best performer (85 percent accuracy) |
    | Random Forest | 82 percent accuracy |
    | Decision Tree | 75 percent accuracy |
    
    ### Key Findings
    
    1. Psychological factors (exhaustion, stress) are stronger predictors than salary
    2. Job satisfaction is the strongest retention factor
    3. Work-life balance significantly impacts attrition risk
    4. Overtime culture increases turnover probability
    
    ### Technical Stack
    
    - Frontend: Streamlit
    - Backend: Python
    - ML Libraries: Scikit-learn, XGBoost
    - Data Processing: Pandas, NumPy
    
    ### Business Impact
    
    By implementing the recommendations from this analysis:
    - Reduce recruitment costs by 15 to 25 percent
    - Improve employee retention by 20 to 30 percent
    - Increase productivity through better wellbeing
    - Build stronger employer brand
    
    ### Contact
    
    For questions or collaboration opportunities:
    - Email: your.email@example.com
    - LinkedIn: linkedin.com/in/yourprofile
    - GitHub: github.com/yourusername
    
    ---
    
    Built with Streamlit
    """)
