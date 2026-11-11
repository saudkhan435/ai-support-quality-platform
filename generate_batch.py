import csv
from pathlib import Path

import httpx

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:3b"

target_intents = {
    "recover_password",
    "track_refund",
    "change_shipping_address"
}

# Select one example for each intent
examples = {}

with open("data/raw/support_tickets.csv", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        intent = row["intent"]

        if intent in target_intents and intent not in examples:
            examples[intent] = row

        if len(examples) == len(target_intents):
            break

# Prepare the results file
output_path = Path("data/evaluation/llm_dev_results.csv")

with open(output_path, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "intent",
            "question",
            "reference_response",
            "generated_response",
            "model"
        ]
    )

    writer.writeheader()

    for intent, example in examples.items():

        question = example["instruction"]
        reference = example["response"]

        prompt = f"""
You are a customer-support assistant.

Customer question:
{question}

Synthetic reference (not verified company policy):
{reference}

Write a short, clear answer using the supplied reference.
Do not invent company policies, URLs, or guaranteed features.
Use cautious wording for unverified procedures.
Keep the answer under 80 words.
"""

        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=180.0
        )

        response.raise_for_status()

        generated = response.json()["response"]

        # Save this question and its generated answer
        writer.writerow({
            "intent": intent,
            "question": question,
            "reference_response": reference,
            "generated_response": generated,
            "model": MODEL
        })

        print(f"Generated response for: {intent}")

print(f"\nResults saved to: {output_path}")