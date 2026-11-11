SELECT
    instruction,
    category,
    intent,
    response

FROM {{ ref('stg_support_tickets') }}

WHERE
    TRIM(instruction) = ''
    OR TRIM(category) = ''
    OR TRIM(intent) = ''
    OR TRIM(response) = ''