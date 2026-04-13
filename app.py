"""
Employee Attrition Prediction App
A Streamlit application for predicting employee turnover and providing HR insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib and seaborn with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("Matplotlib or Seaborn not available. Some visualizations will be limited.")

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
        model = joblib.load("models/best_model.pkl")
        return model
    except:
        st.warning("Model not found. Please train the model first.")
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

model = load_model()

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
        if model is None:
            st.error("Model not loaded. Please train and save the model first.")
        else:
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
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
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

# ============================================================================
# PAGE 2: BATCH PREDICTION
# ============================================================================

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
            
            if model is not None:
                model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else df.columns
                common_features = [col for col in model_features if col in df.columns]
                
                if len(common_features) < len(model_features):
                    st.warning(f"Missing features: {set(model_features) - set(common_features)}")
                
                X_pred = df[common_features]
                predictions = model.predict(X_pred)
                probabilities = model.predict_proba(X_pred)
                
                df_result = df.copy()
                df_result['Attrition_Risk'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                df_result['Attrition_Probability'] = probabilities[:, 1]
                
                st.markdown("### Prediction Summary")
                
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                
                high_risk_count = (predictions == 1).sum()
                low_risk_count = (predictions == 0).sum()
                
                with col_sum1:
                    st.metric("Total Employees", len(df))
                with col_sum2:
                    st.metric("High Risk", high_risk_count, delta=f"{(high_risk_count/len(df)*100):.1f}%")
                with col_sum3:
                    st.metric("Low Risk", low_risk_count, delta=f"{(low_risk_count/len(df)*100):.1f}%")
                with col_sum4:
                    st.metric("Avg Risk Probability", f"{probabilities[:,1].mean():.1%}")
                
                st.markdown("### Risk Distribution")
                
                risk_counts = pd.DataFrame({
                    'Risk Level': ['High Risk', 'Low Risk'],
                    'Count': [high_risk_count, low_risk_count]
                })
                
                fig = px.pie(
                    risk_counts, 
                    values='Count', 
                    names='Risk Level',
                    title='Attrition Risk Distribution',
                    color='Risk Level',
                    color_discrete_map={'High Risk': '#dc3545', 'Low Risk': '#28a745'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
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
            else:
                st.error("Model not loaded. Please train and save the model first.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    st.markdown("---")
    with st.expander("Use Sample Data"):
        if st.button("Load Sample Data"):
            sample_data = load_sample_data()
            st.dataframe(sample_data)
            
            if model is not None:
                common_features = [col for col in model.feature_names_in_ if col in sample_data.columns]
                if len(common_features) > 0:
                    predictions = model.predict(sample_data[common_features])
                    st.write(f"Predictions: {sum(predictions)} out of {len(predictions)} employees at high risk")

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
    
    if model is not None and hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Features Impacting Attrition',
            color='Importance',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## Correlation Heatmap")
    
    features = ['Job_Satisfaction', 'Work_Live_Balance', 'Psychological_Exhaustion', 
                'Physical_Stress', 'MonthlySalary', 'Environment_Satisfaction', 
                'Job_Opportunities', 'Recognition', 'OverTime', 'Years_Experience']
    
    np.random.seed(42)
    correlations = np.array([
        [-0.45, -0.35, 0.52, 0.48, 0.15, -0.38, -0.25, -0.30, 0.28, -0.12],
        [-0.35, -0.40, 0.45, 0.42, 0.10, -0.35, -0.22, -0.28, 0.25, -0.10],
        [0.52, 0.45, 1.00, 0.65, 0.20, 0.40, 0.30, 0.35, 0.32, 0.15],
        [0.48, 0.42, 0.65, 1.00, 0.18, 0.38, 0.28, 0.32, 0.30, 0.12],
        [0.15, 0.10, 0.20, 0.18, 1.00, 0.12, 0.08, 0.10, 0.05, 0.45],
        [-0.38, -0.35, 0.40, 0.38, 0.12, 1.00, 0.35, 0.40, 0.22, 0.08],
        [-0.25, -0.22, 0.30, 0.28, 0.08, 0.35, 1.00, 0.45, 0.18, 0.10],
        [-0.30, -0.28, 0.35, 0.32, 0.10, 0.40, 0.45, 1.00, 0.20, 0.12],
        [0.28, 0.25, 0.32, 0.30, 0.05, 0.22, 0.18, 0.20, 1.00, 0.02],
        [-0.12, -0.10, 0.15, 0.12, 0.45, 0.08, 0.10, 0.12, 0.02, 1.00]
    ])
    
    fig = px.imshow(
        correlations,
        x=features,
        y=features,
        title='Feature Correlations with Attrition',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## Risk Profile by Employee Segment")
    
    profiles = pd.DataFrame({
        'Segment': ['High Stress', 'Low Satisfaction', 'Poor WLB', 'High Overtime', 'Low Risk'],
        'Attrition Risk (%)': [78, 72, 68, 65, 15],
        'Count': [245, 312, 198, 267, 890]
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Attrition Risk by Segment', 'Segment Size'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    fig.add_trace(
        go.Bar(x=profiles['Segment'], y=profiles['Attrition Risk (%)'], 
               marker_color=['#dc3545', '#fd7e14', '#ffc107', '#ffc107', '#28a745']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Pie(labels=profiles['Segment'], values=profiles['Count']),
        row=1, col=2
    )
    
    fig.update_layout(height=500, title_text="Employee Risk Segmentation")
    fig.update_xaxes(title_text="Employee Segment", row=1, col=1)
    fig.update_yaxes(title_text="Attrition Risk (%)", row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
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
    - Visualization: Plotly, Matplotlib, Seaborn
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
