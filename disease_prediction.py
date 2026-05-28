import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)

# ─────────────────────────────────────────
#  Load Dataset
# ─────────────────────────────────────────
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
# 0 = Malignant (Cancer)  |  1 = Benign (No Cancer)

print("=" * 50)
print("   DISEASE PREDICTION — BREAST CANCER")
print("=" * 50)
print(f"\nDataset Shape : {df.shape}")
print(f"Malignant (0) : {(df['target'] == 0).sum()} samples")
print(f"Benign    (1) : {(df['target'] == 1).sum()} samples")

# ─────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────
#  Train 3 Models
# ─────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'                : SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    results[name] = {
        'model': model, 'pred': y_pred,
        'prob': y_prob, 'acc': acc, 'roc': roc
    }

    print(f"\n{'─' * 50}")
    print(f"Model    : {name}")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC  : {roc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=['Malignant', 'Benign']
    ))

# Best model
best = max(results, key=lambda x: results[x]['acc'])
print(f"\n✅ Best Model : {best}  ({results[best]['acc']*100:.2f}% accuracy)")

# ─────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Disease Prediction — Model Evaluation', fontsize=14, fontweight='bold')

# 1. Accuracy bar chart
names = list(results.keys())
accs  = [results[m]['acc'] * 100 for m in names]
bars  = axes[0].bar(names, accs, color=['#3498db', '#2ecc71', '#e74c3c'], width=0.5)
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim([88, 100])
axes[0].tick_params(axis='x', rotation=10)
for bar, val in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 val + 0.1, f'{val:.1f}%',
                 ha='center', fontweight='bold', fontsize=10)

# 2. Confusion matrix (best model)
cm = confusion_matrix(y_test, results[best]['pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
axes[1].set_title(f'Confusion Matrix\n({best})')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('Actual Label')

# 3. ROC curves
for name in names:
    fpr, tpr, _ = roc_curve(y_test, results[name]['prob'])
    axes[2].plot(fpr, tpr,
                 label=f"{name} (AUC = {results[name]['roc']:.3f})",
                 linewidth=2)
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
axes[2].set_title('ROC Curves — All Models')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('disease_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved → disease_prediction_results.png")
