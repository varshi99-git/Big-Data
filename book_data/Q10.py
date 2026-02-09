from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, col, length, avg

# 1. Start Spark Session
spark = SparkSession.builder.appName("BookMetadataAnalysis").getOrCreate()

# 2. Load the Data
rdd = spark.sparkContext.wholeTextFiles("spark/book_data/*.txt")
books_df = rdd.toDF(["file_name", "text"])

# 3. Metadata Extraction
extracted_df = books_df.select(
    col("file_name"),
    regexp_extract("text", r"Title:\s+(.*)", 1).alias("title"),
    regexp_extract("text", r"Release Date:\s+(.*)", 1).alias("release_date"),
    regexp_extract("text", r"Language:\s+(.*)", 1).alias("language"),
    regexp_extract("text", r"Character set encoding:\s+(.*)", 1).alias("encoding")
)

extracted_df.show(truncate=False)

# 4. Analysis
print("Books per Year:")
extracted_df.withColumn("year", regexp_extract("release_date", r"\d{4}", 0)).groupBy("year").count().show()

print("Average Title Length:")
extracted_df.select(avg(length(col("title")))).show()

spark.stop()
