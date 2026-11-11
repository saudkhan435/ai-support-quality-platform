{{ config(materialized='table') }}

SELECT DISTINCT
    instruction,
    category,
    intent

FROM {{ ref('stg_support_tickets') }}