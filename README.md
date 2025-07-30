
# 🫀 Heart Failure Prediction Using Machine Learning

Predicting the likelihood of death in heart failure patients using real clinical data and machine learning models (SVM and ANN).

---

## 📁 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
- **Rows**: 299 patients  
- **Features**: 13 clinical features  
- **Target**: `DEATH_EVENT` (1 = death, 0 = survived)

---

## 🚀 Objective

Build models to:
- Predict patient mortality
- Understand feature influence using visual analysis
- Compare SVM and ANN performance
- Handle class imbalance and optimize model accuracy

---

## 🧪 Technologies Used

- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas, NumPy
- Seaborn, Matplotlib
- Google Colab

---

## 📊 Exploratory Data Analysis (EDA)

- Checked for missing values
- Visualized correlations using heatmaps
- Used boxen plots and swarm plots to analyze feature distribution across death outcomes
- Detected class imbalance in the target variable

---

## 🧠 Models Developed

### 1. Support Vector Machine (SVM)
- Tuned using `GridSearchCV`
- Achieved ~81% accuracy

### 2. Artificial Neural Network (ANN)
- 3 hidden layers with dropout and ReLU activation
- Used early stopping and class weighting
- Final model achieved **85% accuracy**

---

## ⚙️ Model Evaluation

- Metrics: Accuracy, Precision, Recall, F1-score
- **Best Results:**
  ```
  Accuracy:       85%
  Precision (1):  0.70
  Recall (1):     0.84
  F1-score (1):   0.76
  ```

---

## 🧠 Key Learnings

- The importance of handling class imbalance using class weights and SMOTE
- How dropout and early stopping help prevent overfitting in neural networks
- Real-world clinical features can be effectively modeled with ML

---

## 📎 How to Run

1. Open the [Google Colab Notebook](https://colab.research.google.com/drive/17sSitKasq8sPt-Rb3YXEnwQSYedTBJ2p)
2. Upload the dataset or use the UCI link
3. Run the cells step-by-step for EDA, preprocessing, training, and evaluation

---

## 📚 References

- UCI Dataset: [Heart Failure Clinical Records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
- Scikit-learn documentation
- Keras official guides

---

## 👤 Author

**Saumyaketu Chand Gupta**  
LinkedIn: [saumyaketu](https://www.linkedin.com/in/saumyaketu/)  
GitHub: [saumyaketu](https://github.com/Saumyaketu)
