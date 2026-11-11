import duckdb

con = duckdb.connect()

con.sql("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT instruction) AS unique_questions,
        COUNT(*) - COUNT(DISTINCT instruction) AS repeated_questions
    FROM read_csv_auto('data/raw/support_tickets.csv')
""").show()