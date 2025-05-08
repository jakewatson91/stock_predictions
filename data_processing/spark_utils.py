from pyspark.sql import SparkSession
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import unix_timestamp, col, min, max, row_number, to_date
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer

PROJECT_ID='kalshi-456121'
BUCKET_NAME='kalshi-data-lake'
CLUSTER='kalshi-dataproc-cluster'
REGION='us-central1'

# Path to the GCS connector jar
from os.path import expanduser
gcs_connector_path = expanduser("/home/jwatson/gcs-connector-hadoop3-latest.jar") # change depending on where it's stored
# Start a local Spark session with the GCS connector
spark = SparkSession.builder \
    .appName("QueryGCSParquet") \
    .config("spark.jars", gcs_connector_path) \
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
    .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/home/jwatson/cs541/stock_predictions/creds/gcp-sa-key.json") \
    .config("spark.hadoop.fs.gs.project.id", "kalshi-456121") \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .getOrCreate()

def read_parquet():
    # GCS bucket path to the Parquet file or folder
    gcs_path = f"gs://{BUCKET_NAME}/processed_data/"

    # Load the Parquet file from GCS
    data = spark.read.parquet(f"{gcs_path}/*.parquet")
    # data = spark.read.parquet(f"{gcs_path}/part-00000-b3c7923c-db74-44a4-8ad7-37cd92c3a014-c000.snappy.parquet")
    print("Records: ", data.count())
    # df.printSchema()
    # df.show(1)
    return data

def cast_numeric(df, cols):
    # convert each column in `cols` to Spark DoubleType
    for c in cols:
        df = df.withColumn(c, col(c).cast("double"))
    return df 

def normalize_numeric(df, numeric_cols):
    # 1) compute all the means and stddevs in one pass
    stats = df.select(
        *[F.mean(c).alias(f"{c}_mean") for c in numeric_cols],
        *[F.stddev(c).alias(f"{c}_std")   for c in numeric_cols]
    ).first()
    
    # 2) for each column, subtract mean and divide by std (or 1 if std==0)
    for c in numeric_cols:
        mean = stats[f"{c}_mean"]
        std  = stats[f"{c}_std"] or 1.0
        df = df.withColumn(c, (F.col(c) - F.lit(mean)) / F.lit(std))
    return df

def prep_spark(df, numeric_cols):
    df = df.drop("market_type", "open_time", "close_time")
    df = cast_numeric(df, numeric_cols)
    df = normalize_numeric(df, numeric_cols)
    return df 

def top_k_markets_per_day(df, k=100, date_col="created_date", metric_col="daily_total_volume"):
    # define a window partitioned by day, ordered by your metric descending
    w = Window.partitionBy("created_date").orderBy(col(metric_col).desc())

    # add a row_number within each day
    df_ranked = df.withColumn("rn", row_number().over(w))

    # filter to top k and drop the helper columns
    df_topk = (
        df_ranked
          .filter(col("rn") <= k)
          .drop("rn", "day")
    )
    return df_topk
