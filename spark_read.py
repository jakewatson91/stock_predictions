PROJECT_ID='kalshi-456121'
BUCKET_NAME='kalshi-data-lake'
CLUSTER='kalshi-dataproc-cluster'
REGION='us-central1'

from pyspark.sql import SparkSession

def read_parquet(spark):
    # GCS bucket path to the Parquet file or folder
    gcs_path = f"gs://{BUCKET_NAME}/processed_data/"

    # Load the Parquet file from GCS
    df = spark.read.parquet(f"{gcs_path}/*.parquet")

    print("Records: ", df.count())
    df.printSchema()
    df.show(1)
    spark.stop()
    return df