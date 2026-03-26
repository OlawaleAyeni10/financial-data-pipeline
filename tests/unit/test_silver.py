from pyspark.sql import SparkSession
from src.bronze_layer import run_bronze_layer
from src.silver_layer import run_silver_layer

spark = SparkSession.builder \
    .appName("SilverTest") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df_bronze = run_bronze_layer(spark)
df_silver = run_silver_layer(spark, df_bronze)
df_silver.show(5)
print("Silver schema:")
df_silver.printSchema()
spark.stop()