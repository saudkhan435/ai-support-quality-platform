import joblib
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI
from pydantic import BaseModel, field_validator

# Create our API application
app = FastAPI()

# Load our previously trained model
saved_data = joblib.load(
    "artifacts/intent_classifier.joblib"
)

model = saved_data["model"]

# Retrieve the questions used to train the model
training_questions = saved_data["training_questions"]

# Retrieve our trained TF-IDF vectorizer
vectorizer = model.named_steps["tfidf"]

# Convert training questions into numerical vectors
training_vectors = vectorizer.transform(training_questions)

# Prepare a nearest-neighbor similarity search
neighbors = NearestNeighbors(
    n_neighbors=1,
    metric="cosine"
)

neighbors.fit(training_vectors)


# Define the expected request format
class PredictionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:

        # Remove whitespace from the beginning and end
        cleaned_question = value.strip()

        # Reject empty questions
        if not cleaned_question:
            raise ValueError("Question must not be empty")

        return cleaned_question


# Health endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):

    question = [request.question]

    # Predict intent and confidence
    predicted_intent = model.predict(question)[0]

    probabilities = model.predict_proba(question)[0]
    confidence = float(probabilities.max())

    # Convert the customer question into a TF-IDF vector
    question_vector = vectorizer.transform(question)

    # Find its closest training question
    distances, _ = neighbors.kneighbors(question_vector)

    # Convert cosine distance into similarity
    similarity = float(1 - distances[0][0])

    # Our provisional reliability thresholds
    CONFIDENCE_THRESHOLD = 0.70
    SIMILARITY_THRESHOLD = 0.70

    # Both conditions must pass
    if (
        confidence >= CONFIDENCE_THRESHOLD
        and similarity >= SIMILARITY_THRESHOLD
    ):
        decision = "auto_classified"
        needs_human_review = False
    else:
        decision = "needs_human_review"
        needs_human_review = True

    return {
        "predicted_intent": predicted_intent,
        "confidence": confidence,
        "similarity": similarity,
        "decision": decision,
        "needs_human_review": needs_human_review
    }