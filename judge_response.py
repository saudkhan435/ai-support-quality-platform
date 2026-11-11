import csv
import json

import httpx

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Load our existing evaluation data
with open(
    "data/evaluation/llm_dev_results.csv",
    encoding="utf-8",
    newline=""
) as file:
    rows = list(csv.DictReader(file))

# Select the PIN-recovery example
example = next(
    row for row in rows
    if row["intent"] == "recover_password"
)

prompt = f"""
You are evaluating a customer-support AI response.

Customer question:
{example["question"]}

Synthetic reference (NOT verified company policy):
{example["reference_response"]}

Generated response:
{example["generated_response"]}

Known system capabilities:
- No account or profile-management API was accessed.
- No verified PIN-reset procedure is available.
- The assistant cannot confirm that a "Reset PIN" button exists.
- No PIN-reset action has been performed.

Your task:
Identify unsupported factual claims and promises
in the generated response.

CRITICAL:
- Fabricated claims that an action has already been completed.
- Unsupported promises that an action will happen.
- Guaranteed company procedures without verified evidence.
- Fabricated customer-specific information.

MINOR:
- Low-impact unsupported factual statements.

Important rules:
1. Return ONLY exact quotations copied from the generated response.
2. Do not paraphrase or invent claims.
3. Do not list things the response DID NOT say.
4. Distinguish completed actions from promises of future actions.
5. Do not classify ordinary uncertainty disclaimers as unsupported
   merely because they express uncertainty.
6. The synthetic reference is not verified company policy.
7. Do not calculate a score.

Return a JSON object containing exactly:
- critical_unsupported_claims: list of exact quotations
- minor_unsupported_claims: list of exact quotations
"""

response = httpx.post(
    OLLAMA_URL,
    json={
        "model": "qwen2.5:7b",
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0}
    },
    timeout=240.0
)

response.raise_for_status()

# Parse the structured evaluation
result = json.loads(response.json()["response"])

critical = result["critical_unsupported_claims"]
minor = result["minor_unsupported_claims"]

# Verify that every claim is an exact quotation
# from the generated answer, not from the reference.

generated_answer = example["generated_response"]



# Validate the judge's structured output
if not isinstance(critical, list) or not isinstance(minor, list):
    raise ValueError("Judge must return two lists of claims.")

all_claims = critical + minor

invalid_claims = [
    claim
    for claim in all_claims
    if not isinstance(claim, str)
    or not claim.strip()
    or claim not in generated_answer
]
if invalid_claims:
    raise ValueError(
        f"Judge returned claims not found in the generated answer: "
        f"{invalid_claims}"
    )

# Python calculates the score
if critical:
    score = 0
elif minor:
    score = 1
else:
    score = 2

print("Intent:", example["intent"])
print("Critical unsupported claims:", critical)
print("Minor unsupported claims:", minor)
print("Groundedness score:", score)