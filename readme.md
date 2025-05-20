# 💻 Machine Learning Web Application for Price Prediction and Recommendation

This project is a complete end-to-end machine learning pipeline and web application that processes a messy, real-world dataset of computer products scraped from an online marketplace. It handles multilingual data cleaning, price prediction using ML models, clustering for segmentation, and recommendation of similar products. The final output is an interactive app that supports descriptive, predictive, and prescriptive analytics.

---

## 🔍 Objectives

- Clean and preprocess semi-structured, multilingual product data.
- Predict product prices using regression models.
- Enable similarity-based product recommendations.
- Cluster similar products to clarify categories.
- Deploy a user-facing web application with interactive features.
- Recommend similar products based on user input.

---

## 📊 Dataset Overview

- **Rows:** 8,064
- **Columns:** 135
- **Language:** Spanish
- **Types:** Text, numeric, categorical, semi-structured
- **Source:** Online marketplace with laptops, desktops, and components

---

## 🧹 Data Cleaning & Preprocessing

Key tasks:
- Normalized and renamed columns.
- Used regular expressions and helper functions (e.g., `parse_spanish_number`) to clean text.
- Applied domain-specific imputations:
  - **Mode**: For common categorical/binary fields  
  - **Median**: For skewed numeric values  
  - **Domain Inference**: Used related fields to infer missing values  
  - **Special Labels**: Explicitly used 'Missing' or 'None' when appropriate
- Dropped high-missingness or low-value fields.
- Standardized units (e.g., Wh for battery, GB for RAM).

---

## 🛠️ Feature Engineering

- Created derived features:
  - `screen_size_class`, `total_pixel_count`
  - Binary flags for features like Bluetooth, camera, touchscreen
- Parsed:
  - Product title into brand
  - GPU, screen resolution, RAM, storage into structured values
- Grouped:
  - Brands into top and "Other"
  - Product types into high-level categories (e.g., "Laptop – Gaming")

---

## 📈 Exploratory Data Analysis (EDA)

- Key findings:
  - Strong price correlation with RAM, GPU, screen resolution
  - Most common screen sizes: 14", 15.6", and 16"
  - Top brands by count: HP, Lenovo, ASUS
- Visualizations:
  - Brand distribution
  - Missingness matrix
  - Screen size histograms
  - Price distribution
- Feature Decisions:
  - Grouped low-frequency brands
  - Dropped columns with >95% missingness
  - Standardized units and removed outliers

---

## 📦 Machine Learning

- Regression models for price prediction
- Feature importance used for interpretability
- Clustering for product segmentation
- Distance metrics for recommending similar products

---

## 💻 Application Features

- **Descriptive:** 
  - Brand popularity, screen size trends, price distributions
- **Predictive:** 
  - Price estimator based on user inputs
  - Feature importance charts
- **Prescriptive:** 
  - Product recommender with similarity ranking
  - “Better deals” suggestions

---

## 🧠 Lessons Learned

- Real-world data cleaning needs domain-specific inference
- Regex parsing is more robust than naive cleaning
- Binary flags and normalization improve model readiness
- Forward-fill works well in structured product series
- Always drop columns **after** cleaning to prevent data loss

---

## 🚀 Deployment

The app was deployed as a web application with full interactivity. Feedback mechanisms were also implemented to allow iterative improvements.

---

## 📁 Notebooks and Code

See `EDA.ipynb` for the structured notebook used in data cleaning, transformation, and EDA.

---

## 👥 Authors

Oskar Kloczko, Ruben de Juan Grande, Daniel Aguilar, Michail Sifakis, George Vashakidze, Qamar Hussein Aftan

