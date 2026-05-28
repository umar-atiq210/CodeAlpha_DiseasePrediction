# Disease Prediction — CodeAlpha ML Internship Task 4

Predicts breast cancer (Malignant / Benign) using patient tumor data.

## Models Used
- Logistic Regression
- Random Forest
- SVM (Support Vector Machine)

## Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

## Dataset
Breast Cancer Wisconsin — built into Scikit-learn (569 samples, 30 features)

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python disease_prediction.py
```

## Output
- Terminal: accuracy + classification report for all 3 models
- Saved: `disease_prediction_results.png` (bar chart, confusion matrix, ROC curves)
