SELECT
    instruction
FROM {{ ref('stg_support_tickets') }}
GROUP BY instruction
HAVING
    COUNT(DISTINCT intent) > 1
    OR COUNT(DISTINCT category) > 1