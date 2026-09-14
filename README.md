# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn, based on their account details, services, and billing information. Includes exploratory data analysis, model comparison, feature importance analysis, and a deployed interactive prediction app.

**Live Demo:** [https://customer-churn-prediction-tfgquaumsgrlypasxenm4w.streamlit.app/](https://customer-churn-prediction-tfgquaumsgrlypasxenm4w.streamlit.app/)

## Problem Statement

Customer churn directly impacts revenue for subscription-based businesses. This project analyzes a telecom customer dataset to identify key churn drivers and build a model that flags at-risk customers, enabling proactive retention efforts.

## Dataset

Telco Customer Churn dataset (Kaggle) — 7,043 customer records with 21 features including tenure, contract type, internet service, monthly/total charges, and churn status.

## Approach

1. **Data Cleaning:** Handled missing values in `TotalCharges` (11 rows, all customers with 0 tenure — filled with 0).
2. **EDA:** Explored churn patterns across contract type, internet service, and tenure.
3. **Preprocessing:** One-hot encoded 15 categorical features, resulting in 30 model-ready features.
4. **Modeling:** Trained and compared three classifiers — Logistic Regression, Random Forest, and XGBoost.
5. **Evaluation:** Compared models on accuracy, precision, recall, and F1 — prioritizing recall, since missing an at-risk customer is costlier to the business than a false alarm.
6. **Deployment:** Built an interactive Streamlit app for real-time churn risk prediction.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 |
| Random Forest | 0.762 | 0.539 | 0.725 | 0.618 |
| **XGBoost (selected)** | 0.756 | 0.528 | **0.751** | **0.620** |

**Why XGBoost:** While Logistic Regression had the highest accuracy, XGBoost achieved the best recall and F1 score — correctly identifying 75% of actual churners versus 57% for Logistic Regression. In a churn use case, catching more at-risk customers (even at the cost of some false positives) is more valuable to the business than raw accuracy, since a missed churner is a lost customer while a false alarm just means an unnecessary retention offer.

## Key Insight

Contract type is by far the strongest churn predictor — customers on month-to-month contracts, especially those with fiber optic internet, show significantly higher churn risk. The business should prioritize retention offers (e.g., contract upgrade incentives) for this specific segment.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
