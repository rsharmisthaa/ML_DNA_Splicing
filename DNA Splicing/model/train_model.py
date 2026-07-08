import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD DATA ----------------
df = pd.read_csv(os.path.join(BASE_DIR, "../data/dna.csv"))

print("=" * 60)
print("DATASET INFO")
print("=" * 60)

print(df.head())
print("\nShape:", df.shape)

print("\nClass Distribution")
print(df.iloc[:, -1].value_counts())

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nFeatures:", X.shape[1])

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ---------------- MODELS ----------------
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    ),
    "SVM": SVC(
        probability=True,
        kernel="rbf",
        random_state=42,
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),
}

# ---------------- TRAIN ----------------
for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")

    preds = model.predict(X_test)

    print("\nFirst 20 Predictions")
    print(preds[:20])

    print("\nFirst 20 Actual Labels")
    print(y_test.values[:20])

    print("\nPrediction Counts")
    print(pd.Series(preds).value_counts())

    print("\nActual Counts")
    print(y_test.value_counts())

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report")
    print(classification_report(y_test, preds))

# ---------------- SAVE ----------------
joblib.dump(models, os.path.join(BASE_DIR, "models.pkl"))
joblib.dump(X_test, os.path.join(BASE_DIR, "X_test.pkl"))
joblib.dump(y_test, os.path.join(BASE_DIR, "y_test.pkl"))

print("\nModels saved successfully.")