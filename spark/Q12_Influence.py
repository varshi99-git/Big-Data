import sys
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, count, lit, desc
from pyspark.sql.types import StringType, IntegerType, StructType, StructField

def extract_metadata(text):
    # Default values
    author = None
    year = None

    # Regex for Author
    author_match = re.search(r"Author:\s+(.+)", text)
    if author_match:
        author = author_match.group(1).strip()

    # Regex for Release Date (Extracting Year only)
    # Looks for 4 digits in a line starting with "Release Date"
    date_match = re.search(r"Release Date:.*?(\d{4})", text)
    if date_match:
        try:
            year = int(date_match.group(1))
        except:
            year = None

    if author and year:
        return (author, year)
    return None

if __name__ == "__main__":
    spark = SparkSession.builder.appName("AuthorInfluenceNetwork").getOrCreate()

    # 1. Load Data
    rdd = spark.sparkContext.wholeTextFiles("file:///mnt/c/Users/saiva/Final_Submission/spark/book_data/*.txt")

    # 2. Extract Metadata (Author, Year)
    # We use flatMap to filter out files where metadata couldn't be found (returns empty list)
    processed_rdd = rdd.map(lambda x: extract_metadata(x[1])).filter(lambda x: x is not None)

    # Create DataFrame
    schema = StructType([
        StructField("author", StringType(), True),
        StructField("year", IntegerType(), True)
    ])

    # Distinct to avoid counting the same book twice if duplicates exist
    # We keep (Author, Year) pairs. If an author published multiple books in 1900, 
    # we can keep them to weigh influence, or distinct them. 
    # Let's keep unique (Author, Year) entries to simplify "did they publish then?"
    meta_df = spark.createDataFrame(processed_rdd, schema).distinct()

    # 3. Create Network Edges
    # Self-join to find pairs
    # Condition: Author A (influencer) released a book BEFORE Author B (influenced),
    # but within X years (e.g., 5 years).

    X = 5 # Time window

    # Aliasing for self-join
    df1 = meta_df.alias("df1") # Potential Influencer
    df2 = meta_df.alias("df2") # Potential Influenced

    edges_df = df1.join(df2, 
        (col("df1.author") != col("df2.author")) &      # Different authors
        (col("df2.year") >= col("df1.year")) &          # df1 came before or same year as df2
        (col("df2.year") <= col("df1.year") + X)        # But within X years
    ).select(
        col("df1.author").alias("influencer"), 
        col("df2.author").alias("influenced")
    ).distinct() # A link exists if ANY book satisfies this, don't count multiple books multiple times

    # Cache edges for analysis
    edges_df.cache()

    # 4. Analysis: Out-Degree (Who influenced the most people?)
    out_degree = edges_df.groupBy("influencer") \
                         .agg(count("influenced").alias("out_degree_score")) \
                         .orderBy(desc("out_degree_score")) \
                         .limit(5)

    # 5. Analysis: In-Degree (Who was influenced by the most people?)
    in_degree = edges_df.groupBy("influenced") \
                        .agg(count("influencer").alias("in_degree_score")) \
                        .orderBy(desc("in_degree_score")) \
                        .limit(5)

    print("------------------------------------------------")
    print(f"Top 5 Influential Authors (Out-Degree, Window={X} yrs):")
    out_degree.show(truncate=False)

    print(f"Top 5 Influenced Authors (In-Degree, Window={X} yrs):")
    in_degree.show(truncate=False)
    print("------------------------------------------------")

    spark.stop()
