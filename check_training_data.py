import duckdb

con = duckdb.connect("support_analytics/dev.duckdb", read_only=True)

result = con.sql("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT instruction) AS unique_questions
    FROM main.ml_training_data
""")

result.show()

con.close()