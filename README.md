# 🚨 FinOps Fraud Detection ML API

A real-time fraud detection API powered by a pre-trained **XGBoost** machine learning model. Built with **FastAPI**, it predicts the likelihood of a financial transaction being fraudulent based on historical behavior, transaction metadata, and encoded categorical features.

---

## 📌 Features

- 🔍 Fraud detection using XGBoost
- 📈 Behavioral profiling over 7-day transaction history
- 🧠 In-memory user behavior tracking
- 🧾 One-hot encoding for categorical fields (e.g., method, location, device)
- ⚡ High-performance API using FastAPI
- 🔐 Flag simulation for potentially suspicious transactions
- 🗂️ Automatically updates categories based on existing transaction dataset

---

## 📁 Project Structure

```bash
.
├── app.py                 # Main FastAPI application
├── fraud_model.pkl        # Trained XGBoost model
├── transactions.csv       # Dataset used for initializing category values & statistics
├── requirements.txt       # Dependencies (example list below)
└── README.md              # Project documentation
---
## ⚙️ Set-up Instructions

git clone https://github.com/MilanMishra19/finops.git    # Clone the  repository
pip install -r requirements.txt                          # Install dependencies
Ensure fraud_model.pkl (trained XGBoost model)
is available in the root directory.
uvicorn app:app --reload                                 # Start the Server
The API will be available at: **http://localhost:8000**

---
## 🚀 API Endpoints
GET /
Returns a basic message confirming that the API is live.
{
  "message": "Fraud Detection API is running!"
}

---
## Backend Repository
https://github.com/MilanMishra19/finopsbackend
## Frontend Repository
https://github.com/MilanMishra19/finopsfrontend
