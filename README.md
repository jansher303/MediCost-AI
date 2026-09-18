# MediCost-AI

### Medical Insurance Cost Prediction using Machine Learning

MediCost-AI is a machine learning project that predicts medical insurance costs based on personal and health-related information such as age, gender, BMI, number of children, smoking status, and region.

The project uses **Multiple Linear Regression** and the **Medical Cost Personal Dataset**. It covers the complete machine learning workflow, including data exploration, preprocessing, exploratory data analysis, model training, evaluation, and deployment through a Streamlit web application.

---

## 📌 Project Overview

Medical insurance charges can vary significantly depending on factors such as age, BMI, smoking habits, number of children, and geographical region.

The goal of this project is to build a machine learning model that can learn the relationship between these factors and medical insurance charges, and then predict the expected insurance cost for a new individual.

---

## 🎯 Objectives

* Explore and understand the insurance dataset
* Clean and preprocess the raw data
* Convert categorical variables into numerical values
* Perform exploratory data analysis
* Train a Multiple Linear Regression model
* Evaluate the model using regression metrics
* Build a simple prediction application using Streamlit
* Make the project reproducible through GitHub

---

## 📊 Dataset

The project uses the **Medical Cost Personal Dataset**, containing information about individuals and their medical insurance charges.

### Original Features

| Feature    | Description                   |
| ---------- | ----------------------------- |
| `age`      | Age of the individual         |
| `sex`      | Gender of the individual      |
| `bmi`      | Body Mass Index               |
| `children` | Number of children/dependents |
| `smoker`   | Smoking status                |
| `region`   | Residential region            |
| `charges`  | Medical insurance charges     |

The dataset contains **1,338 records** and **7 original columns**.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Checked the dataset for missing values.
2. Checked for duplicate records.
3. Removed duplicate records where necessary.
4. Converted categorical variables into numerical values.
5. Encoded `sex` and `smoker` using binary encoding.
6. Applied one-hot encoding to the `region` feature.
7. Used `charges` as the target variable.

After preprocessing, the dataset contained **9 columns**.

### Encoded Features

```text
age
sex
bmi
children
smoker
region_northwest
region_southeast
region_southwest
charges
```

`region_northeast` was used as the reference category because one-hot encoding was performed with `drop_first=True`.

---

## 📈 Exploratory Data Analysis

Several visualizations were used to understand relationships within the dataset, including:

* Age vs. Medical Charges
* BMI vs. Medical Charges
* Smoker vs. Medical Charges
* Feature Correlation Heatmap
* Actual vs. Predicted Charges

### Key Observations

* Medical charges generally increase with age.
* Smoking status shows a strong relationship with medical charges.
* BMI has a noticeable but less consistent relationship with charges.
* Several input features contribute to the variation in medical insurance costs.

---

## 🤖 Machine Learning Model

### Multiple Linear Regression

The project uses **Multiple Linear Regression** from Scikit-learn.

The target variable is:

```text
charges
```

The model uses the following input features:

```text
age
sex
bmi
children
smoker
region_northwest
region_southeast
region_southwest
```

### Train/Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

The train/test split used:

```python
test_size=0.2
random_state=42
```

---

## 📊 Model Evaluation

The trained model was evaluated using four regression metrics.

| Metric       |            Result |
| ------------ | ----------------: |
| **MAE**      |      **4,177.05** |
| **MSE**      | **35,478,020.68** |
| **RMSE**     |      **5,956.34** |
| **R² Score** |        **0.8069** |

### What the results mean

The model achieved an **R² score of 0.8069**, meaning the model explains approximately **80.69% of the variation in medical insurance charges in the test set**.

**MAE (4,177.05)** represents the average absolute difference between the actual and predicted charges.

**RMSE (5,956.34)** represents the prediction error in the same units as the target variable and gives greater weight to larger errors.

---

## 🌐 Streamlit Application

A Streamlit web application was developed to allow users to enter individual information and receive a predicted medical insurance cost.

### User Inputs

The application accepts:

* Age
* Gender
* BMI
* Number of Children
* Smoking Status
* Region

After entering the information, the trained machine learning model generates the predicted medical insurance cost.

### Application Preview

> Screenshots will be added here.

---

## 🔄 Machine Learning Workflow

```text
Medical Insurance Dataset
          ↓
Data Exploration
          ↓
Data Cleaning
          ↓
Categorical Encoding
          ↓
Exploratory Data Analysis
          ↓
Train/Test Split
          ↓
Multiple Linear Regression
          ↓
Model Evaluation
          ↓
Streamlit Application
          ↓
Medical Insurance Cost Prediction
```

---

## 📂 Project Structure

```text
MediCost-AI/
│
├── app.py
│       └── Streamlit prediction application
│
├── model.pkl
│       └── Trained Multiple Linear Regression model
│
├── requirements.txt
│       └── Required Python dependencies
│
├── README.md
│       └── Project documentation
│
└── screenshots/
        ├── app-interface.png
        ├── prediction-test-1.png
        ├── prediction-test-2.png
        └── actual-vs-predicted.png
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jansher303/MediCost-AI.git
```

### 2. Navigate to the Project

```bash
cd MediCost-AI
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open automatically in your web browser.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Streamlit**
* **Git**
* **GitHub**
* **Google Colab**

---

## 📚 Machine Learning Concepts Used

* Data Cleaning
* Data Preprocessing
* Categorical Encoding
* One-Hot Encoding
* Exploratory Data Analysis
* Feature/Target Separation
* Train/Test Split
* Multiple Linear Regression
* Model Prediction
* MAE
* MSE
* RMSE
* R² Score

---

## 🚀 Future Improvements

Possible improvements for future versions include:

* Comparing Multiple Linear Regression with other regression algorithms
* Hyperparameter tuning
* Feature engineering
* Improved UI/UX
* Interactive data visualizations
* Cloud deployment
* Model performance comparison
* Automated model retraining

---

## ⚠️ Disclaimer

This project is created for **educational and demonstration purposes**. The predicted insurance cost should not be considered an actual insurance quotation or financial/medical advice.

---

## 👨‍💻 Author

### Jan Sher

**BS Computer Science**
University of Management and Technology, Lahore

GitHub: [jansher303](https://github.com/jansher303)

---

## ⭐ Acknowledgment

This project was developed as part of practical learning in **Machine Learning and Artificial Intelligence**, covering the complete workflow from raw data preprocessing to model deployment.
