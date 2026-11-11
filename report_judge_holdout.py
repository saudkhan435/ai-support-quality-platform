from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.core.datasets import MulticlassClassification
from evidently.presets import ClassificationPreset

# Load Qwen's holdout evaluation results
df = pd.read_csv(
    "data/evaluation/judge_holdout_results.csv"
)

# Keep only structurally valid evaluations
valid = df[df["status"] == "valid"].copy()

if valid.empty:
    raise ValueError("No valid judge evaluations found.")

# Treat groundedness scores as classification labels
valid["gold_score"] = valid["gold_score"].astype(str)
valid["judge_score"] = valid["judge_score"].astype(str)

# Tell Evidently which column is ground truth
# and which column contains Qwen's predictions
dataset = Dataset.from_pandas(
    valid[["gold_score", "judge_score"]],
    data_definition=DataDefinition(
        classification=[
            MulticlassClassification(
                target="gold_score",
                prediction_labels="judge_score"
            )
        ]
    )
)

# Evaluate classification performance
report = Report([ClassificationPreset()])

snapshot = report.run(dataset)

output = Path("data/evaluation/judge_holdout_report.html")
snapshot.save_html(str(output))

# Print our main results
correct = (
    valid["gold_score"] == valid["judge_score"]
).sum()

accuracy = correct / len(valid)

print(f"Valid evaluations: {len(valid)}/{len(df)}")
print(f"Judge accuracy: {accuracy:.2%}")
print(f"Report saved to: {output}")