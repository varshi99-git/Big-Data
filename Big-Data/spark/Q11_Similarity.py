from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, desc, lit, expr
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.sql.types import StringType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("GutenbergAnalysis") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("="*50)
print("PART 1: AUTHOR INFLUENCE NETWORK")
print("="*50)

# ---------------------------------------------------------
# Step 1.1: Preprocessing (Simulating Book Metadata)
# ---------------------------------------------------------
# Simulating the raw text rdd/dataframe from Gutenberg headers
data_influence = [
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
df_meta = spark.createDataFrame(data_influence, ["raw_text"])

# Extract Author and Year using Regex
# Pattern captures: "by (Author Name). Released: (Year)"
df_clean = df_meta.withColumn("author", regexp_extract(col("raw_text"), r"by ([\w\.\s]+)\.", 1)) \
                  .withColumn("year", regexp_extract(col("raw_text"), r"Released: (\d{4})", 1).cast("int"))

print(">>> Extracted Metadata:")
df_clean.show(5, truncate=False)

# ---------------------------------------------------------
# Step 1.2: Influence Network Construction
# ---------------------------------------------------------
# Logic: Author A influences Author B if A published a book 
# within X years BEFORE B (inclusive).
X = 5 

# Self-join the dataframe
# Left side = Influencer (src), Right side = Influenced (dst)
edges = df_clean.alias("src").join(
    df_clean.alias("dst"),
    (col("src.author") != col("dst.author")) &         # Distinct authors
    (col("src.year") <= col("dst.year")) &             # Time must move forward
    ((col("dst.year") - col("src.year")) <= X)         # Within X years
).select(
    col("src.author").alias("influencer"), 
    col("dst.author").alias("influenced")
).distinct()

print(f">>> Influence Network Edges (Window={X} years):")
edges.show(5, truncate=False)

# ---------------------------------------------------------
# Step 1.3: Analysis (In-Degree & Out-Degree)
# ---------------------------------------------------------
# In-Degree: How many people influenced me?
in_degree = edges.groupBy("influenced").count().withColumnRenamed("count", "in_degree")

# Out-Degree: How many people did I influence?
out_degree = edges.groupBy("influencer").count().withColumnRenamed("count", "out_degree")

print(">>> Top Authors by In-Degree (Most Influenced):")
in_degree.orderBy(desc("in_degree")).show(5)

print(">>> Top Authors by Out-Degree (Most Influential):")
out_degree.orderBy(desc("out_degree")).show(5)


print("\n" + "="*50)
print("PART 2: TF-IDF & COSINE SIMILARITY")
print("="*50)

# ---------------------------------------------------------
# Step 2.1: Preprocessing (Simulating Book Content)
# ---------------------------------------------------------
# Simulating 4 books. 
# 10.txt and 12.txt are similar (Space topic).
# 11.txt is different (Cooking).
data_tfidf = [
    ("10.txt", "astronomy space stars planets galaxy universe"),
    ("11.txt", "cooking recipes food kitchen dinner ingredients"),
    ("12.txt", "space galaxy stars astronomy telescope hubble"),
    ("13.txt", "deep space nine universe planets solar system")
]
df_books = spark.createDataFrame(data_tfidf, ["file_name", "text"])

# Tokenize
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df_words = tokenizer.transform(df_books)

# Remove Stop Words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
df_filtered = remover.transform(df_words)

# ---------------------------------------------------------
# Step 2.2: TF-IDF Calculation
# ---------------------------------------------------------
# TF (HashingTF)
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=100)
featurizedData = hashingTF.transform(df_filtered)

# IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Normalize vectors (L2 norm) so that Dot Product equals Cosine Similarity
# Cosine(A, B) = (A . B) / (|A| * |B|)
# If |A| and |B| are 1 (normalized), then Cosine(A, B) = A . B
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=2.0)
df_normalized = normalizer.transform(rescaledData)

print(">>> TF-IDF Vectors (Normalized):")
df_normalized.select("file_name", "normFeatures").show(truncate=False)

# ---------------------------------------------------------
# Step 2.3: Book Similarity (Target: 10.txt)
# ---------------------------------------------------------
target_file = "10.txt"

# Extract the vector for the target book
target_vector_row = df_normalized.filter(col("file_name") == target_file).select("normFeatures").first()

if target_vector_row:
    target_vector = target_vector_row["normFeatures"]
    
    # Define a UDF to calculate dot product (Cosine Similarity)
    # Note: Spark's ML vectors support dot product natively in Python
    def cosine_sim(v):
        return float(v.dot(target_vector))
    
    from pyspark.sql.functions import udf
    from pyspark.sql.types import FloatType
    
    cosine_udf = udf(cosine_sim, FloatType())
    
    # Calculate similarity for all books against the target
    df_similarity = df_normalized.withColumn("similarity", cosine_udf(col("normFeatures")))
    
    print(f">>> Top Similar Books to {target_file}:")
    df_similarity.filter(col("file_name") != target_file) \
                 .select("file_name", "similarity") \
                 .orderBy(desc("similarity")) \
                 .show()
else:
    print(f"Target file {target_file} not found.")

spark.stop()
