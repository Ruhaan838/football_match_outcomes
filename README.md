# ⚽ Football Model Trainer
### Predicting Football Match Outcomes Using Team Statistics

Welcome to the **Football Model Trainer**, a simple desktop application for training and evaluating machine learning models using a graphical interface. This project supports both **regression** and **binary classification** tasks using real football-related datasets.

Whether you're a student, a data enthusiast, or someone learning machine learning, this app gives you hands-on experience with model training, evaluation, visualization, and saving/loading models — all without needing to write a single line of code.

---

## 📦 Features

- ✅ Easy-to-use GUI built with **Tkinter**
- ⚙️ Supports **Regression** and **Binary Classification**
- 📊 Plots **confusion matrix**, **bar charts**, and **classification error distribution**
- 🧠 Uses **Decision Tree Regressor** and **Classifier**
- 💾 Save and Load models using `joblib`
- 📈 Evaluation metrics: Accuracy, Precision, Recall, F1-score
- 🔄 Verbose mode for logging training details

---

## 🖥 Technologies Used

- **Python 3**
- **Tkinter** (for GUI)
- **Scikit-learn** (for models and metrics)
- **Seaborn** and **Matplotlib** (for visualization)
- **Pandas** and **NumPy** (for data handling)
- **Joblib** (for model saving and loading)

---

## 🚀 How It Works

### 1. Choose the Model Type
Select either:
- `regression`
- `b_classification` (binary classification)

### 2. Train the Model
Click the **Train Model** button. The app will:
- Load and split the dataset
- Train the selected model
- Show training progress in the log panel

### 3. Evaluate the Model
Click **Evaluate Model** to see:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Optional visualizations

> If a model is loaded from disk, it will evaluate on test data directly without retraining.

### 4. Plot Visuals
- **Confusion Matrix** – for classification
- **True/False Error Distribution** – for classification
- **Regression Prediction Curve** – for regression

### 5. Save or Load Model
You can:
- Save the trained model (`weights/m_classify.joblib`)
- Load an existing model and evaluate it without retraining

---

## 🧪 Example

- Select model type: `regression`
- Click **Train Model**
- Click **Evaluate Model**
- Click **Plot graph**

You will see training logs, a performance report, and plots.

---