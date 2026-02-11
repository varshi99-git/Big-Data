from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, desc, count

# Initialize Spark
spark = SparkSession.builder \
    .appName("AuthorInfluence") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# --- DATA PREPARATION ---
data = [
    ("The Great Gatsby by F. Scott Fitzgerald. Released: 1925",),
    ("This Side of Paradise by F. Scott Fitzgerald. Released: 1920",),
    ("The Sun Also Rises by Ernest Hemingway. Released: 1926",),
    ("A Farewell to Arms by Ernest Hemingway. Released: 1929",),
    ("The Sound and the Fury by William Faulkner. Released: 1929",),
    ("As I Lay Dying by William Faulkner. Released: 1930",),
    ("Mrs. Dalloway by Virginia Woolf. Released: 1925",),
    ("To the Lighthouse by Virginia Woolf. Released: 1927",),
    ("Ulysses by James Joyce. Released: 1922",),
    ("Portrait of the Artist by James Joyce. Released: 1916",)
]
df_raw = spark.createDataFrame(data, ["text"])

# Extract Author and Year
df = df_raw.withColumn("author", regexp_extract("text", r"by ([\w\.\s]+)\.", 1)) \
           .withColumn("year", regexp_extract("text", r"Released: (\d{4})", 1).cast("int"))

# --- INFLUENCE NETWORK CONSTRUCTION ---
X = 5  # Time Window

# Self-Join to find edges
edges = df.alias("src").join(df.alias("dst"), 
    (col("src.author") != col("dst.author")) & 
    (col("src.year") <= col("dst.year")) & 
    ((col("dst.year") - col("src.year")) <= X)
).select(
    col("src.author").alias("influencer"), 
    col("dst.author").alias("influenced")
).distinct()

# --- ANALYSIS & OUTPUT ---

# 1. Out-Degree (Most Influential)
out_degree = edges.groupBy("influencer").agg(count("influenced").alias("out_degree_score")) \
                  .orderBy(desc("out_degree_score"))

print("-" * 50)
print("Top 5 Influential Authors (Out-Degree, Window=5 yrs):")
out_degree.show(5, truncate=False)

# 2. In-Degree (Most Influenced)
in_degree = edges.groupBy("influenced").agg(count("influencer").alias("in_degree_score")) \
                 .orderBy(desc("in_degree_score"))

print("-" * 50)
print("Top 5 Influenced Authors (In-Degree, Window=5 yrs):")
in_degree.show(5, truncate=False)
print("-" * 50)

spark.stop()
