FROM apache/airflow:slim-3.4.0

RUN python -m venv /home/airflow/dbt-venv && \
    /home/airflow/dbt-venv/bin/pip install --no-cache-dir dbt-duckdb==1.11.1