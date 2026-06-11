# 🏠 Smart Tehran House Price Prediction (2025)

This project is an interactive web application that uses Machine Learning algorithms and data extracted from **Divar** to estimate house prices in various neighborhoods of Tehran for the year 2025.

🚀 **Check out the live app here:** [Tehran House Price Predictor](https://tehran-house-price-prediction-6aubby8w6aifarcuth6bxn.streamlit.app/)

---

## ✨ Features
- **Smart Ensemble Model:** Utilizes an advanced `VotingRegressor` combining LightGBM, CatBoost, and Random Forest to maximize prediction accuracy.
- **Production-Ready & Lightweight:** Fully refactored to remove heavy experimental dependencies (like PyCaret), ensuring fast inference and smooth deployment.
- **Custom User Interface:** Built with Streamlit, featuring a tailored RTL layout and Vazirmatn font for a native Persian user experience.
- **Robust Data Handling:** Implements Target Encoding for neighborhood mapping, complete with a fallback mechanism for handling unseen or unknown addresses.

---

## 🛠️ Tech Stack & Libraries
The following tools and libraries were used in this project:
- **Python 3.x**
- **Streamlit** (Frontend & UI)
- **Scikit-Learn** (Base algorithms and Voting Regressor)
- **LightGBM** & **CatBoost** (Advanced Gradient Boosting models)
- **Joblib** (For efficient model serialization and loading)
- **Pandas** & **Numpy** (Data manipulation and preprocessing)

---

## 📊 Dataset
The dataset consists of real estate listings scraped from the **Divar** platform in Tehran. The data preprocessing pipeline included the following steps:
1. Cleaning and handling missing values.
2. Removing outliers using the Quantile method for both house area and price.
3. Encoding neighborhood addresses into meaningful numerical values based on the average historical price of each area.
4. Exporting the neighborhood mapping to a JSON file for use during the inference phase.

---

## 📁 Project Structure

```text
├── data/
│   ├── tehranhouses.csv       # Raw/Cleaned dataset
│   └── address_mapping.json   # Mapping of neighborhoods to target-encoded values
├── models/
│   └── tehran_house_price_pipeline.pkl  # Trained serialized model
├── src/
│   ├── notebook.ipynb         # EDA, Model training, and evaluation
│   └── app.py                 # Main Streamlit application
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
