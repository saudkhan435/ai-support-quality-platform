from datetime import datetime, timezone
from pathlib import Path
import subprocess

from airflow.sdk import dag, task


DBT = "/home/airflow/dbt-venv/bin/dbt"
PROJECT = "/opt/project/support_analytics"
PROFILES = "/opt/airflow/dbt_profiles"


@dag(
    dag_id="support_data_pipeline",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["data-engineering", "dbt"],
)
def support_data_pipeline():

    @task
    def check_dataset():
        path = Path("/opt/project/data/raw/support_tickets.csv")
        if not path.is_file():
            raise FileNotFoundError(f"Dataset missing: {path}")
        print("Dataset found!")

    @task
    def run_dbt_models():
        subprocess.run(
            [
                DBT, "run",
                "--select", "+ml_training_data",
                "--profiles-dir", PROFILES,
            ],
            cwd=PROJECT,
            check=True,
        )

    @task
    def run_dbt_tests():
        subprocess.run(
            [
                DBT, "test",
                "--select", "+ml_training_data",
                "--profiles-dir", PROFILES,
            ],
            cwd=PROJECT,
            check=True,
        )

    check_dataset() >> run_dbt_models() >> run_dbt_tests()


support_data_pipeline()