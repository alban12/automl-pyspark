import pyspark.sql.functions as F

def with_greeting(df):
    return df.withColumn("greeting", F.lit("hello!")) 

def remove_non_word_characters(col):
    return F.regexp_replace(col, "[^\\w\\s]+", "")
