"""
Train and evaluate land-cover classification models.
Dataset: synthetic educational sample; replace with verified real satellite data for research use.
Run: python train_model.py
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "Satellite_Land_Cover_700_Sample.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "NDVI"]
TARGET = "Land_Cover"

df = pd.read_csv(DATA)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": Pipeline([("scaler", StandardScaler()),
                     ("model", SVC(kernel="rbf", probability=True, random_state=42))]),
    "KNN": Pipeline([("scaler", StandardScaler()),
                     ("model", KNeighborsClassifier(n_neighbors=5))]),
}

scores = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    scores.append({"Algorithm": name, "Accuracy": accuracy})
    fitted[name] = model
    print(f"{name}: {accuracy:.4f}")

scores_df = pd.DataFrame(scores).sort_values("Accuracy", ascending=False)
scores_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
best_name = scores_df.iloc[0]["Algorithm"]
best_model = fitted[best_name]
joblib.dump(best_model, MODEL_DIR / "land_cover_model.joblib")

best_prediction = best_model.predict(X_test)
pd.DataFrame(classification_report(
    y_test, best_prediction, output_dict=True, zero_division=0
)).transpose().to_csv(RESULTS_DIR / "classification_report.csv")

print("\\nBest model:", best_name)
print("Test accuracy:", accuracy_score(y_test, best_prediction))
print("\\nClassification report:")
print(classification_report(y_test, best_prediction, zero_division=0))
print("\\nSaved model to:", MODEL_DIR / "land_cover_model.joblib")
