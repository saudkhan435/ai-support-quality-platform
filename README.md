# AI Support Quality Platform

An end-to-end Data Science and MLOps project for processing customer-support data, predicting customer intents, generating AI-assisted responses, and evaluating response quality.

## 1. Problem Statement

Customer-support systems need to understand customer requests and provide reliable answers.

However, a language model can generate convincing responses that contain invented company procedures, fabricated actions, or unsupported promises.

This project addresses two problems:

- **Intent classification:** Identify what a customer is asking about.
- **Response quality evaluation:** Detect potentially unreliable AI-generated customer-support answers.

The platform combines data engineering, machine learning, LLM evaluation, API development, and MLOps practices.

## 2. Architecture

The project uses the following technologies:

| Component | Technology | Purpose |
|---|---|---|
| Data orchestration | Apache Airflow | Execute data-processing workflows |
| Data storage | DuckDB | Local analytical database |
| Data transformation | dbt | Prepare and validate training data |
| Intent classification | scikit-learn | Predict customer-support intents |
| Experiment tracking | MLflow | Record model experiments and metrics |
| Response generation | Ollama + Llama 3.2 | Generate customer-support responses locally |
| LLM evaluation | Qwen 2.5 + Python | Identify potentially unsupported claims |
| Evaluation reporting | Evidently | Produce quality metrics and reports |
| Model serving | FastAPI | Expose intent predictions through an API |
| Containerization | Docker | Package and run the API |
| CI/CD | GitHub Actions | Run tests, publish images, and deploy after successful checks |
| Container registry | GitHub Container Registry (GHCR) | Store versioned Docker images |
| Cloud deployment | Azure Container Apps | Host the containerized classifier API |

**The intent-classification API is deployed to Azure Container Apps through an automated GitHub Actions CI/CD pipeline.** The data pipeline and local LLM evaluation remain separate from the cloud deployment.

## 3. Intent Classification

The classifier uses TF-IDF text features and Logistic Regression to classify customer-support questions into 27 intent categories.

The training workflow includes:

- Stratified training, validation, and test splits
- A DummyClassifier baseline
- Accuracy and macro F1 evaluation
- MLflow experiment tracking
- Confidence and text-similarity checks for routing uncertain predictions to human review

### Initial validation results

| Metric | Result |
|---|---|
| Accuracy | 99.14% |
| Macro F1 | 99.08% |
| Dummy baseline accuracy | 4.06% |

**Limitation:** The dataset contains synthetic, template-driven examples. These scores may overestimate real-world generalization.

## 4. Local LLM Response Evaluation

Llama 3.2 generates customer-support answers using synthetic reference responses.

Qwen 2.5 acts as an experimental LLM judge, identifying potentially unsupported claims.

Python validates the returned evidence and calculates groundedness scores.

### Groundedness scoring

- **0 — Ungrounded:** Significant fabricated action or unsupported operational promise.
- **1 — Partially grounded:** Minor unsupported claims.
- **2 — Grounded:** No unsupported substantive claims identified.
- **Invalid:** The judge returned evidence that failed validation.

Flagged or invalid evaluations require human review.

### Initial evaluation results

- Calibration benchmark: 6/6 correct
- Synthetic holdout benchmark: 6/6 correct
- Generated responses: 2/3 structurally valid evaluations
- Responses flagged for human review: 3/3

These results are exploratory. Small synthetic benchmarks do not establish production reliability, and the judge can misclassify claims even when its output is structurally valid.

## 5. Running the Project

### Intent Classification API

The FastAPI service can be run locally through Docker Compose:

```powershell
docker compose -f compose.api.yaml up --build -d
```

Health endpoint:

`http://localhost:8001/health`

Prediction endpoint:

`POST http://localhost:8001/predict`

The current Dockerized API serves intent classification. Local LLM generation and evaluation are separate experimental workflows.

### Local LLM Evaluation

Requirements:

- Python 3.11
- Ollama
- `llama3.2:3b`
- `qwen2.5:7b`

Install evaluation dependencies:

```powershell
python -m pip install -r requirements-evaluation.txt
```

Ensure Ollama is accessible at `http://127.0.0.1:11434`.

Run the evaluation pipeline:

```powershell
python generate_batch.py
python evaluate_generated_responses.py
python report_llm_final.py
```

The integrated report is saved to:

`data/evaluation/llm_final_report.html`

The judge's standalone calibration and holdout experiments are available through `evaluate_judge_benchmark.py` and `evaluate_judge_holdout.py`.

## 6. Current Limitations

- The customer-support dataset is synthetic.
- Reference responses are not verified company policies.
- LLM judge scores are provisional and can contain errors.
- No live customer account, refund, or shipping systems are connected.
- LLM evaluation currently runs locally rather than as part of the deployed API.
## 7. Cloud Deployment and CI/CD

The intent-classification API is deployed to Microsoft
Azure Container Apps in the Germany West Central region.

### Continuous Integration

GitHub Actions automatically:
- Runs FastAPI unit tests.
- Builds the Docker image.
- Starts the container and runs integration tests.

### Continuous Deployment

After successful tests:
- GitHub Actions builds and publishes a versioned
  Docker image to GitHub Container Registry (GHCR).
- GitHub Actions authenticates to Azure using
  OpenID Connect (OIDC).
- Azure Container Apps deploys the new image.

The deployed API exposes:
- GET /health
- POST /predict
- GET /docs

Azure Container Apps uses the Consumption plan with
0 minimum replicas and 1 maximum replica.

The cloud deployment serves the trained scikit-learn
intent classifier. Airflow, dbt, MLflow and local
LLM evaluation are not deployed as cloud services.

### Cost Considerations

Deployment was demonstrated using Azure Free Trial
credits. Actual cloud availability depends on the
Azure subscription and its billing status.

The API is an educational prototype, not a
production customer-support service.


## Dataset Attribution

This project uses the Bitext Customer Support LLM Chatbot
Training Dataset, containing 26,872 synthetic
customer-support question/answer examples.

Source:
https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

Dataset provider: Bitext Innovations

Dataset license: CDLA-Sharing-1.0
https://cdla.dev/sharing-1-0/

The dataset is used for educational experimentation.
Its synthetic responses are not verified company policies.