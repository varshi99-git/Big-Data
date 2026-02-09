import sys
import re
import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, count, lit, sum as _sum, sqrt, collect_list
from pyspark.sql.types import ArrayType, StringType, DoubleType, MapType

def clean_text(text):
    # Remove header/footer (simplified for this example)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize (split by whitespace)
    words = text.split()
    # Remove stop words (simple list)
    stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it', 'with', 'as', 'was', 'on', 'at', 'by', 'be', 'this', 'are', 'or', 'an'])
    return [w for w in words if w not in stop_words and len(w) > 2]

if __name__ == "__main__":
    spark = SparkSession.builder.appName("BookSimilarityTFIDF").getOrCreate()

    # 1. Load Data
    rdd = spark.sparkContext.wholeTextFiles("file:///mnt/c/Users/saiva/Final_Submission/spark/book_data/*.txt")
    books_df = rdd.toDF(["file_path", "raw_text"])

    # Extract just the filename (e.g., "10.txt")
    extract_filename = udf(lambda path: path.split("/")[-1], StringType())
    books_df = books_df.withColumn("file_name", extract_filename("file_path"))

    # 2. Preprocessing
    clean_udf = udf(clean_text, ArrayType(StringType()))
    books_df = books_df.withColumn("words", clean_udf("raw_text"))

    # Explode to get one row per word per book
    words_df = books_df.select("file_name", explode("words").alias("word"))

    # 3. TF Calculation (Term Frequency)
    # Count occurrences of each word in each book
    tf_df = words_df.groupBy("file_name", "word").count().withColumnRenamed("count", "term_count")

    # Calculate total words per book
    total_words_df = words_df.groupBy("file_name").count().withColumnRenamed("count", "total_words")

    # TF = (count of word in doc) / (total words in doc)
    tf_df = tf_df.join(total_words_df, "file_name")
    tf_df = tf_df.withColumn("tf", col("term_count") / col("total_words"))

    # 4. IDF Calculation (Inverse Document Frequency)
    total_docs = books_df.count()

    # Count how many docs contain each word
    doc_freq_df = tf_df.groupBy("word").count().withColumnRenamed("count", "doc_freq")

    # IDF = log(total_docs / (doc_freq + 1))
    idf_df = doc_freq_df.withColumn("idf", udf(lambda df: math.log((total_docs + 1) / (df + 1)), DoubleType())(col("doc_freq")))

    # 5. TF-IDF Calculation
    tfidf_df = tf_df.join(idf_df, "word")
    tfidf_df = tfidf_df.withColumn("tfidf", col("tf") * col("idf"))

    # Cache this dataframe as we will use it multiple times
    tfidf_df.cache()

    # 6. Book Similarity (Cosine Similarity)
    # Vectorize: For each book, collect map of {word: tfidf}
    # Note: In production, use MLlib. We do this manually per assignment instructions.

    # Let's filter for a specific target book to compare against, e.g., "10.txt" (The Bible)
    target_book = "10.txt"

    # Get vector for target book
    target_vector = tfidf_df.filter(col("file_name") == target_book).select("word", "tfidf").rdd.collectAsMap()

    # Broadcast the target vector to all nodes
    target_broadcast = spark.sparkContext.broadcast(target_vector)

    # Define Cosine Similarity UDF
    def calculate_cosine(book_words, book_tfidfs):
        target_vec = target_broadcast.value

        dot_product = 0.0
        norm_a = 0.0
        norm_b = 0.0

        # Calculate dot product and norm for the current book
        for i in range(len(book_words)):
            word = book_words[i]
            val = book_tfidfs[i]
            norm_a += val ** 2

            if word in target_vec:
                dot_product += val * target_vec[word]

        # Calculate norm for target book (pre-calculated or done here)
        norm_b = sum([v**2 for v in target_vec.values()])

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))

    # Prepare data for comparison: group by filename
    # We need lists of words and values to pass to UDF
    book_vectors = tfidf_df.groupBy("file_name").agg(
        collect_list("word").alias("word_list"),
        collect_list("tfidf").alias("tfidf_list")
    )

    # Apply similarity function
    similarity_udf = udf(calculate_cosine, DoubleType())
    similarity_df = book_vectors.withColumn("similarity", similarity_udf("word_list", "tfidf_list"))

    # Sort by similarity descending
    result = similarity_df.select("file_name", "similarity") \
                          .filter(col("file_name") != target_book) \
                          .orderBy(col("similarity").desc()) \
                          .limit(5)

    print(f"\nTop 5 books similar to {target_book}:")
    result.show(truncate=False)

    spark.stop()
