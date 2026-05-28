# Placement Predictor (Machine Learning)

An end-to-end Machine Learning classification project that predicts whether a student will secure a job placement based on their academic performance (**CGPA**) and cognitive ability (**IQ**).

---

## 📌 Project Overview

This repository contains a predictive analysis system using supervised machine learning. Given historical data of students' performance metrics and their final placement status, the classification model learns the underlying boundary to categorize future students into two distinct classes:
* **1**: Placed
* **0**: Not Placed

---

## 📊 Dataset & Features

The model utilizes a tabular dataset containing the following features:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **CGPA** | Float | Cumulative Grade Point Average of the student. |
| **IQ** | Integer/Float | Intelligence Quotient score. |
| **Placement** | Binary (0/1) | **Target Variable** (0 = Not Placed, 1 = Placed). |

---

## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn
* **Environment:** Jupyter Notebook / VS Code

---

## 🚀 Workflow Pipeline

1.  **Data Preprocessing:** Handling missing values, exploring data distributions, and feature scaling (standardization/normalization).
2.  **Exploratory Data Analysis (EDA):** Visualizing the decision boundary space using scatter plots ($CGPA$ vs. $IQ$ colored by placement status).
3.  **Train-Test Split:** Partitioning the dataset into training and testing sets to evaluate generalization performance.
4.  **Model Training:** Training classification algorithms (e.g., Logistic Regression, Support Vector Machines, or Random Forest).
5.  **Evaluation:** Assessing the model using metrics like **Accuracy**, **Precision**, **Recall**, and plotting the **Confusion Matrix** or **ROC Curve**.

---

