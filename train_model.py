import csv
from pathlib import Path

import duckdb
import joblib
import mlflow
import mlflow.sklearn

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

con = duckdb.connect(
    "support_analytics/dev.duckdb",
    read_only=True
)

rows = con.sql("""
    SELECT instruction, intent
    FROM main.ml_training_data
""").fetchall()

con.close()

X = [row[0] for row in rows]
y = [row[1] for row in rows]

# First split: 80% development, 20% test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.2,
    random_state=42,
    stratify=y_dev
)

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))



# Build our ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=2
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# Train the model
model.fit(X_train, y_train)



# Create the artifacts folder if it doesn't exist
Path("artifacts").mkdir(exist_ok=True)

# Save the trained model and training questions
joblib.dump(
    {
        "model": model,
        "training_questions": X_train
    },
    "artifacts/intent_classifier.joblib"
)

print("Model saved successfully!")

# Save validation data separately

validation_path = Path("data/evaluation/validation.csv")

validation_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(validation_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # Column names
    writer.writerow(["question", "intent"])

    # Save questions and their correct labels
    writer.writerows(zip(X_val, y_val))

print(f"Saved {len(X_val)} validation examples.")




# Evaluate on validation data
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average="macro")

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")



baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_val)

print("\nBaseline Results:")
print(
    "Accuracy:",
    round(accuracy_score(y_val, baseline_pred), 4)
)
print(
    "Macro F1:",
    round(f1_score(y_val, baseline_pred, average="macro"), 4)
)

# Store experiment results locally
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("support_intent_classification")

# Experiment 1: Baseline
# Experiment 1: Baseline
with mlflow.start_run(run_name="dummy_baseline"):
    mlflow.log_param("model", "DummyClassifier")

    mlflow.log_metric(
        "val_accuracy",
        accuracy_score(y_val, baseline_pred)
    )

    mlflow.log_metric(
        "val_macro_f1",
        f1_score(y_val, baseline_pred, average="macro")
    )

# Experiment 2: Trained classifier
with mlflow.start_run(run_name="tfidf_unigrams"):
    mlflow.log_params({
        "model": "LogisticRegression",
        "ngram_range": "1,1",
        "min_df": 2,
        "max_iter": 1000
    })

    mlflow.log_metrics({
    "val_accuracy": accuracy,
    "val_macro_f1": macro_f1,
    "feature_count": len(
        model.named_steps["tfidf"].vocabulary_
    )
})
    mlflow.sklearn.log_model(
        sk_model=model,
        name="intent_classifier"
    )

