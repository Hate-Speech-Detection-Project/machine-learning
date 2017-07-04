import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras

file = '../../data/10000/train.csv'
data = pd.read_csv(file, sep=',')
data.sort_values('created')

# Connect to the database.
try:
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='admin'")
except:
    print("Cannot connect to database")
    sys.exit(0)
print("Connected to database")

# Create a temporary view in the database with the comments in the dataset.
with conn.cursor() as cursor:
  cursor.execute(
    """
    CREATE TEMPORARY VIEW temp_comments AS
    SELECT * FROM comments
    WHERE cid IN (%s)
    ORDER BY created
    """ % (','.join([str(row['cid']) for index, row in data.iterrows()]))
  )

# Define the feature queries.
queries = [(
  # Time since last comment (by anyone) on same article.
  'time_since_last_comment',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.url = inside.url
    AND outside.created > inside.created
  )
  FROM temp_comments AS outside
  ORDER BY created
  """
),(
  # Time since last comment (by same user) on same article.
  'time_since_last_comment_same_user',
  """
  SELECT outside.created - (
    SELECT COALESCE(MAX(inside.created), outside.created)
    FROM comments AS inside
    WHERE outside.url = inside.url
    AND outside.created > inside.created
    AND outside.uid = inside.uid
  )
  FROM temp_comments AS outside
  ORDER BY created
  """
),(
  # Number of comments by user at the time of writing the comment.
  'number_of_comments_by_user',
  """
  SELECT (
  	SELECT COUNT(1)
    FROM comments AS inside
    WHERE inside.uid = outside.uid
    AND inside.created < outside.created
  )
  FROM temp_comments AS outside
  ORDER BY created
  """
)]

# Execute the queries and cache the results.
def execute_query(query):
  with conn.cursor() as cursor:
    cursor.execute(query)
    return [x[0] for x in cursor.fetchall()]
results = [execute_query(query[1]) for query in queries]

# Create a nice dataframe with all the data.
data_result = pd.DataFrame(results)
data_result = data_result.transpose()
data_result.columns = [query[0] for query in queries]
data = data.join(data_result)

# Output file.
data.to_csv(file + '.augmented.csv', encoding='utf-8')