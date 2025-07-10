# 🫀 Heart Disease Prediction Using Machine Learning

Predict the presence of heart disease based on patient health metrics using machine learning algorithms.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Model Performance](#model-performance)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

---

## 🧠 Project Overview

This project focuses on building a machine learning pipeline to detect heart disease in patients based on various clinical parameters. Early prediction of heart disease can save lives and reduce healthcare costs. The model is trained using popular algorithms and deployed via a simple web app using Flask.

---

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- **Samples**: 303 patient records
- **Features**:
  - Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol, Fasting Blood Sugar
  - Max Heart Rate, Exercise-induced Angina, ST Depression, Number of Vessels Colored, Thalassemia
- **Target**: `target` (0 = No Heart Disease, 1 = Heart Disease)

---

## 💻 Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `scikit-learn`
  - Deployment: `Flask`, `pickle`

---

## 🔁 Project Workflow

1. **Data Preprocessing**
   - Null check, scaling, and encoding of categorical variables
2. **Exploratory Data Analysis**
   - Correlation matrix, feature distribution, class balance
3. **Model Training**
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
4. **Model Evaluation**
   - Confusion matrix, precision, recall, F1-score
5. **Model Deployment**
   - Save best model using `pickle`
   - Create web app using Flask

---

## 📈 Model Performance

| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | 85%      |
| Random Forest          | **90%**  |
| Support Vector Machine | 87%      |

✅ **Random Forest** was selected as the final model for deployment.

---

## 🚀 How to Run

### 🛠️ 1. Clone the repository
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
# Heart-Disease-Prediction
