{{ config(materialized='view') }}

SELECT
    TRIM(flags) AS flags,
    TRIM(instruction) AS instruction,
    TRIM(category) AS category,
    TRIM(intent) AS intent,
    TRIM(response) AS response
FROM read_csv_auto('../data/raw/support_tickets.csv')