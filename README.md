# Sowing Success: How Machine Learning Helps Farmers

## 🌾 Project Overview
This project aims to help farmers optimize their crop yields by identifying the most important soil metrics for crop selection. Measuring soil condition (Nitrogen, Phosphorous, Potassium, and pH) can be expensive; therefore, this analysis identifies which single metric provides the best predictive power to reduce testing costs.

## 📊 Dataset
The dataset `soil_measures.csv` contains:
- **N**: Nitrogen content ratio in the soil.
- **P**: Phosphorous content ratio in the soil.
- **K**: Potassium content ratio in the soil.
- **pH**: pH value of the soil.
- **crop**: The target variable (categorical) representing the optimal crop for the soil conditions.

## 🛠️ Methodology
1. **Exploratory Data Analysis**: Checked for missing values and verified unique crop types.
2. **Model Selection**: Used a **Multi-class Logistic Regression** model.
3. **Feature Importance**: Trained individual models for each soil metric (N, P, K, pH) to determine which one yields the highest **Weighted F1-Score**.
4. **Optimization**: Increased `max_iter` to ensure model convergence.

## 🚀 Key Results
The analysis revealed that **Potassium (K)** is the single most important feature for predicting the best crop type in this dataset, achieving the highest predictive performance.

## 📂 Technologies Used
- Python
- Pandas
- Scikit-learn (LogisticRegression, train_test_split, metrics)
