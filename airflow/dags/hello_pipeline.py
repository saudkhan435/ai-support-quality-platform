from datetime import datetime, timezone
from airflow.sdk import dag, task


@dag(
    dag_id="hello_pipeline",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["learning"],
)
def hello_pipeline():

    @task
    def check_input():
        print("Input check completed!")

    @task
    def confirm_completion():
        print("Workflow completed successfully!")

    check_input() >> confirm_completion()


hello_pipeline()