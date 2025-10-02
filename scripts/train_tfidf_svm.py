import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # assure que le dossier existe

data = pd.read_csv("data/all_tickets_processed_improved_v3.csv")

print(data.head())

X, y = data["Document"], data["Topic_group"]

# ===  Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===  Pipeline TF-IDF + SVM calibré ===
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("svc", CalibratedClassifierCV(LinearSVC(), cv=5))
])

# ===  Entraînement ===
pipeline.fit(X_train, y_train)

# === Évaluation ===
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print("✅ Accuracy:", acc)
print("✅ F1-score:", f1)

# === Log dans MLflow ===
with mlflow.start_run():
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("classifier", "LinearSVC + calibration")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(pipeline, "tfidf_svm_model")
    mlflow.log_artifact("outputs/confusion_matrix.png")
    mlflow.log_artifact("outputs/class_distribution.png")
    mlflow.log_artifact("outputs/top_words_class0.png")
# ===  Visualisations ===


## Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=y.unique())
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=y.unique(), yticklabels=y.unique())
plt.xlabel("Prédictions")
plt.ylabel("Vrais labels")
plt.title("Matrice de confusion - TFIDF + SVM")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()  

## Distribution des classes (réel vs prédit)
plt.figure(figsize=(8,5))
pd.Series(y_test).value_counts().sort_index().plot(kind="bar", alpha=0.6, label="Réel")
pd.Series(y_pred).value_counts().sort_index().plot(kind="bar", alpha=0.6, label="Prédit", color="orange")
plt.legend()
plt.title("Distribution des classes : Réel vs Prédit")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
plt.close()

## Top mots discriminants (classe 0)
tfidf = pipeline.named_steps["tfidf"]
feature_names = tfidf.get_feature_names_out()
#coef = pipeline.named_steps["svc"].base_estimator.coef_

svc_estimator = pipeline.named_steps["svc"].base_estimator if hasattr(pipeline.named_steps["svc"], "base_estimator") else pipeline.named_steps["svc"].calibrated_classifiers_[0].estimator

coef = svc_estimator.coef_

top_idx = coef[0].argsort()[-20:]
plt.figure(figsize=(8,6))
plt.barh(range(len(top_idx)), coef[0][top_idx])
plt.yticks(range(len(top_idx)), feature_names[top_idx])
plt.title("Top 20 mots discriminants (classe 0)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_words_class0.png"))
plt.close()