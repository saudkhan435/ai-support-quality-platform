import duckdb

con = duckdb.connect(
    "support_analytics/dev.duckdb",
    read_only=True
)

con.sql("""
    SELECT
        intent,
        COUNT(*) AS total_examples
    FROM main.ml_training_data
    GROUP BY intent
    ORDER BY total_examples DESC
""").show(max_rows=30)

con.close()