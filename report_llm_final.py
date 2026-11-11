import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import WordCount, Contains
from evidently.presets import TextEvals, DataSummaryPreset

# Load Llama's generated responses
responses = pd.read_csv(
    "data/evaluation/llm_dev_results.csv"
)

# Load Qwen's groundedness evaluations
evaluations = pd.read_csv(
    "data/evaluation/llm_groundedness_results.csv"
)

# Combine generation and evaluation results
df = responses.merge(
    evaluations[
        [
            "intent",
            "groundedness_score",
            "status",
            "needs_human_review"
        ]
    ],
    on="intent",
    how="left",
    validate="one_to_one"
)

if df["status"].isna().any():
    raise ValueError("Missing evaluations for some responses.")

# Keep invalid evaluations separate from scored responses
df["groundedness_result"] = df.apply(
    lambda row: "invalid"
    if row["status"] != "valid"
    else str(int(row["groundedness_score"])),
    axis=1
)

# Prepare categorical review labels
df["review_decision"] = df["needs_human_review"].map(
    {True: "review", False: "no_review"}
)

# Add Evidently text descriptors
dataset = Dataset.from_pandas(
    df,
    data_definition=DataDefinition(),
    descriptors=[
        WordCount(
            "generated_response",
            alias="word_count"
        ),
        Contains(
            "generated_response",
            items=["I've checked", "I'll let you know"],
            case_sensitive=False,
            mode="any",
            alias="possible_unsupported_action"
        )
    ]
)

# Combine text quality and evaluation summaries
report = Report([
    TextEvals(),
    DataSummaryPreset(
        columns=[
            "groundedness_result",
            "status",
            "review_decision"
        ]
    )
])

snapshot = report.run(dataset)

output = "data/evaluation/llm_final_report.html"
snapshot.save_html(output)

# Display a compact summary
print("Total responses:", len(df))
print("Valid evaluations:", (df["status"] == "valid").sum())
print("Invalid evaluations:", (df["status"] == "invalid").sum())
print("Human reviews:", (df["review_decision"] == "review").sum())

print("\nGroundedness results:")
print(df["groundedness_result"].value_counts().to_string())

print(f"\nFinal Evidently report saved to: {output}")