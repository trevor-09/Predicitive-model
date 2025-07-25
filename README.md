# 🫀 Heart Failure Prediction Web App

This project is part of the **DevTown Predictive Modelling Bootcamp**, where we trained a machine learning model to predict the risk of death due to heart failure and deployed it using **Flask** with a clean **custom CSS frontend**.

---

## 💡 Overview

- 📊 **Dataset**: Heart Failure Clinical Records Dataset
- 🧠 **Model**: Random Forest Classifier
- 🎯 **Goal**: Predict whether a patient is likely to die based on clinical inputs.
- 🌐 **Deployment**: Flask web app with custom HTML and CSS UI

---

## 📁 Project Structure
HeartFailurePredictor/
│
├── model.pkl # Trained machine learning model
├── app.py # Flask backend
├── templates/
│ └── index.html # Custom styled frontend with input form
└── model_training.ipynb # Jupyter Notebook for model training (optional)


---

## 🚀 Getting Started

### ✅ Prerequisites

Install the required Python libraries:

```bash
pip install flask numpy scikit-learn

▶️ Running the App Locally
Clone this repository

Make sure model.pkl and app.py are in the root folder

Place index.html inside a folder named templates/

Run the app:

python app.py

🧪 Input Fields (on the web app)

| Field                    | Description                          |
| ------------------------ | ------------------------------------ |
| Age                      | Patient's age                        |
| Anaemia                  | 1 if the patient has anaemia, else 0 |
| Creatinine Phosphokinase | CPK enzyme level                     |
| Diabetes                 | 1 if diabetic                        |
| Ejection Fraction        | % of blood leaving the heart         |
| High BP                  | 1 if high blood pressure             |
| Platelets                | Platelet count                       |
| Serum Creatinine         | Blood creatinine level               |
| Serum Sodium             | Blood sodium level                   |
| Sex                      | 1 = Male, 0 = Female                 |
| Smoking                  | 1 if smoker                          |
| Time                     | Follow-up period (in days)           |




