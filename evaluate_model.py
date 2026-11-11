import joblib

# Load our previously trained model
saved_data = joblib.load(
    "artifacts/intent_classifier.joblib"
)

model = saved_data["model"]

# Test one customer question
question = ["How can I delete my account?"]

prediction = model.predict(question)
probabilities = model.predict_proba(question)

confidence = probabilities.max()

print("Question:", question[0])
print("Predicted intent:", prediction[0])
print(f"Confidence: {confidence:.2%}")

import csv

# Load our out-of-scope questions
with open("data/evaluation/ood_dev.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    ood_questions = [row["question"] for row in reader]

# Generate predictions without retraining
ood_probabilities = model.predict_proba(ood_questions)

threshold = 0.70
rejected_count = 0

for question, probs in zip(ood_questions, ood_probabilities):

    confidence = probs.max()

    if confidence < threshold:
        rejected_count += 1

# Calculate rejection rate
rejection_rate = rejected_count / len(ood_questions)

print("\nOOD Evaluation Results")
print("Total questions:", len(ood_questions))
print("Rejected questions:", rejected_count)
print(f"OOD Rejection Rate: {rejection_rate:.2%}")

from sklearn.neighbors import NearestNeighbors

# Retrieve training questions from our saved file
training_questions = saved_data["training_questions"]

# Retrieve the trained TF-IDF vectorizer
vectorizer = model.named_steps["tfidf"]

# Convert training and OOD questions into numerical vectors
train_vectors = vectorizer.transform(training_questions)
ood_vectors = vectorizer.transform(ood_questions)

# Find the closest training question for each OOD question
neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine"
)

neighbors.fit(train_vectors)

distances, _ = neighbors.kneighbors(ood_vectors)

# Convert cosine distance to similarity
ood_similarities = 1 - distances[:, 0]

# Apply both conditions
confidence_threshold = 0.70
similarity_threshold = 0.70

confidences = ood_probabilities.max(axis=1)

accepted = (
    (confidences >= confidence_threshold)
    & (ood_similarities >= similarity_threshold)
)

rejected = (~accepted).sum()

print("\nCombined Evaluation Results")
print("Total OOD questions:", len(ood_questions))
print("Rejected questions:", rejected)
print(f"OOD Rejection Rate: {rejected / len(ood_questions):.2%}")

# Load saved validation data
with open("data/evaluation/validation.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    validation_rows = list(reader)

val_questions = [row["question"] for row in validation_rows]
val_labels = [row["intent"] for row in validation_rows]

# Predict intents and confidence scores
val_predictions = model.predict(val_questions)
val_probabilities = model.predict_proba(val_questions)
val_confidences = val_probabilities.max(axis=1)

# Calculate similarity to training questions
val_vectors = vectorizer.transform(val_questions)

val_distances, _ = neighbors.kneighbors(val_vectors)
val_similarities = 1 - val_distances[:, 0]

# Compare the two acceptance methods
for name, accepted in [
    (
        "Confidence only",
        val_confidences >= confidence_threshold
    ),
    (
        "Confidence + similarity",
        (val_confidences >= confidence_threshold)
        & (val_similarities >= similarity_threshold)
    )
]:
    accepted_count = accepted.sum()

    correct_count = sum(
        prediction == actual
        for prediction, actual, keep in zip(
            val_predictions, val_labels, accepted
        )
        if keep
    )

    coverage = accepted_count / len(val_questions)

    accepted_accuracy = (
        correct_count / accepted_count
        if accepted_count > 0 else 0
    )

    print(f"\nValidation Method: {name}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Accepted Accuracy: {accepted_accuracy:.2%}")
    print(f"Human Reviews: {len(val_questions) - accepted_count}")