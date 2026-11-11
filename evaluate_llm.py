import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import WordCount, Contains
from evidently.presets import TextEvals

# 1. Load our generated responses
df = pd.read_csv("data/evaluation/llm_dev_results.csv")

print("Total responses:", len(df))

# 2. Check for missing or empty answers
empty = df["generated_response"].fillna("").str.strip().eq("").sum()
print("Empty responses:", empty)

# 3. Define Evidently's text evaluations
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

# 4. Display row-level results
results = dataset.as_dataframe()

print(
    results[
        ["intent", "word_count", "possible_unsupported_action"]
    ].to_string(index=False)
)

# 5. Generate an Evidently HTML report
report = Report([TextEvals()])
snapshot = report.run(dataset)

snapshot.save_html("data/evaluation/llm_dev_report.html")

print("\nEvidently report saved successfully.")