#Employee Attrition Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.ai/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-red.svg)](https://scikit-learn.org/)

> **Predicting employee turnover using machine learning to help HR teams take proactive retention actions.**

## Overview

Employee attrition (turnover) is a critical challenge for organizations, leading to increased recruitment costs, loss of institutional knowledge, and decreased productivity. This project analyzes **Saudi employee data** to identify key factors driving attrition and builds predictive models to flag at-risk employees.

**Key Questions Answered:**
- What factors most influence employee attrition?
- Can we predict which employees are likely to leave?
- What actionable insights can HR teams implement?

## Business Impact

| Metric | Insight |
|--------|---------|
| **Top Attrition Drivers** | Psychological exhaustion, physical stress, overtime, job dissatisfaction |
| **Retention Factors** | Job satisfaction, work-life balance, environment satisfaction |
| **High-Risk Groups** | Employees with high job titles, high stress levels, frequent overtime |

> **Key Finding:** Psychological factors (stress, burnout) are stronger predictors of attrition than salary. HR should focus on well-being programs, not just compensation.

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Evaluation** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

## Exploratory Data Analysis (EDA)

### Missing Values Analysis
![Missing Values](images/missing_values.png)
> *All features had 2-5% missing values, handled via imputation.*

### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)
> *Job satisfaction and work-life balance show strong negative correlation with attrition.*

### Target Distribution
![Target Distribution](images/target_distribution.png)
> *Dataset is imbalanced (58% No Attrition, 42% Attrition). Addressed using class_weight.*

## Models & Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | 0.85 | 0.84 | 0.83 | 0.83 |
| Random Forest | 0.82 | 0.81 | 0.80 | 0.80 |
| Decision Tree | 0.75 | 0.74 | 0.73 | 0.73 |

> XGBoost performed best with balanced precision-recall trade-off.

### Feature Importance (XGBoost)
![Feature Importance](images/feature_importance.png)
> *Top predictors: JobTitle, Psychological Exhaustion, MonthlySalary, Physical Stress.*

## Key Insights for HR

### Red Flags (Attrition Drivers)
| Feature | Impact | Action Item |
|---------|--------|--------------|
| Psychological Exhaustion |  High | Implement mental health programs, reduce workload |
| Physical Stress | High | Ergonomic improvements, regular breaks |
| Overtime | Medium | Cap overtime hours, hire more staff |
| Job Dissatisfaction | Medium | Regular 1-on-1 check-ins, career development |

### Green Flags (Retention Factors)
| Feature | Impact | Action Item |
|---------|--------|--------------|
| Job Satisfaction | Strong | Recognition programs, growth opportunities |
| Work-Life Balance | Strong | Flexible hours, remote work options |
| Environment Satisfaction | Strong | Improve office culture, team building |

## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/employee-attrition-prediction.git
cd employee-attrition-prediction
