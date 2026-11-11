import httpx
import csv

# Ollama API address
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Read one customer-support example from our dataset
with open(
    "data/raw/support_tickets.csv",
    "r",
    encoding="utf-8",
    newline=""
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        if row["intent"] == "recover_password":
            customer_question = row["instruction"]
            reference_response = row["response"]
            break


# Build the prompt using dataset information
prompt = f"""
You are a customer-support assistant.

Customer question:
{customer_question}

Synthetic reference response:
{reference_response}

Instructions:
Write a short, clear response using the reference as an example.
The reference is synthetic and its procedures are not verified.
Do not invent company policies, URLs, or guaranteed features.
Use conditional wording for unverified instructions.
Keep your answer under 80 words.
"""
# Information sent to Ollama
payload = {
    "model": "llama3.2:3b",
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0
    }
}

# Send HTTP request to Ollama
response = httpx.post(
    OLLAMA_URL,
    json=payload,
    timeout=180.0
)

# Check whether the request succeeded
response.raise_for_status()

# Extract generated text
generated_answer = response.json()["response"]

print("Generated customer-support response:")
print(generated_answer)