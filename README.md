# 💧 Water Quality Prediction System

## 📌 Project Overview

This project predicts whether water is **safe (potable)** or **unsafe (non-potable)** using Machine Learning.

It uses key water parameters:

* pH
* Total Dissolved Solids (TDS)
* Turbidity
* temperature

The system is designed to work with:

* 📊 Existing dataset (for training)
* 🔌 Real-time sensor data using Arduino (future integration)

---

## 🎯 Objectives

* Build a Machine Learning model for water quality prediction
* Use low-cost sensors for real-time data collection
* Provide a simple user interface for prediction
* Replace dataset with real sensor data in future

---

## 🧠 Tech Stack

| Component            | Technology         |
| -------------------- | ------------------ |
| Programming Language | Python             |
| ML Library           | Scikit-learn       |
| Data Handling        | Pandas, NumPy      |
| Model Saving         | Joblib             |
| UI                   | Tkinter            |
| Hardware             | Arduino Uno        |
| Sensors              | pH, TDS, Turbidity |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```
git clone <your-repo-link>
cd WATER_POTABILITY_PROJECT
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

---

### 3️⃣ Activate Virtual Environment

#### Windows:

```
venv\Scripts\activate
```

#### Linux/Mac:

```
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### 🔹 Train the Model

```
cd model
python train_model.py
```

---

### 🔹 Run the UI

```
cd ../app
python ui.py
```

## 📊 Model Details

* Algorithm: Random Forest Classifier
* Input Features:

  * pH
  * TDS (Solids)
  * Turbidity
* Output:

  * 1 → Safe (Potable)
  * 0 → Unsafe (Non-potable)

---

## ⚠️ Notes

* Virtual environment (`venv/`) is not included in GitHub
* Use `requirements.txt` to install dependencies
* Dataset can be replaced with real sensor data

---

## ⭐ Contribution

Feel free to fork and improve this project!

---
