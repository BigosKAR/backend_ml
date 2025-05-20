# Machine Learning Web Application for Price Prediction and Recommendation

This project is a complete end-to-end machine learning pipeline and web application that processes a messy, real-world dataset of computer products scraped from an online marketplace. It handles multilingual data cleaning, price prediction using ML models, clustering for segmentation, and recommendation of similar products. The final output is an interactive app that supports descriptive, predictive, and prescriptive analytics.

---

## 🔍 Objectives

- Clean and preprocess semi-structured, multilingual product data.
- Predict product prices using regression models.
- Enable similarity-based product recommendations.
- Cluster products into meaningful groups.
- Deploy a user-facing web application with interactive features.

---

## 📊 Dataset Overview

- **Rows:** 8,064
- **Columns:** 135
- **Language:** Spanish
- **Types:** Text, numeric, categorical, semi-structured
- **Source:** Online marketplace of electronic products

---

## 🧹 Data Cleaning & Preprocessing

Key tasks:
- Normalized and renamed columns.
- Used regular expressions and helper functions (e.g., `parse_spanish_number`) to clean text.
- Applied domain-specific imputations:
  - Median for numeric values.
  - Mode for binary/categorical.
  - Context-aware forward-fill (e.g., within product series).
- Dropped high-missingness or low-value fields.
- Standardized units (e.g., Wh for battery, GB for RAM).

---

## 🛠️ Feature Engineering

- Created new features:
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

---

## 📦 Machine Learning

- Regression models for price prediction
- Feature importance used for interpretability
- Clustering for product segmentation
- Cosine similarity and distance metrics for recommending similar products

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

- Regex-based parsing outperforms naive cleaning.
- Forward-fill is effective for structured product series.
- Binary flags improve model training readiness.
- Domain expertise is critical for imputing missing values.
- Track column drops carefully to avoid unintentional data loss.

---

## 🚀 Deployment

The app was deployed as a web application with full interactivity. Feedback mechanisms were also implemented to allow iterative improvements.

---

## 📁 Notebooks and Code

See `3_0_EDA_Structure_Template.ipynb` for the structured notebook used in data cleaning, transformation, and EDA.

---

## 👥 Authors

Oskar Kloczco, Ruben de Juan Grande, Daniel Aguilar, Michail Sifakis, George Vashakidze, Qamar Hussein

