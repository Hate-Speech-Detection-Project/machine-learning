import psycopg2
import psycopg2.extras
import sys
import requests
import cProfile

# Connect to the database.
try:
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='admin'")
except:
    print("Cannot connect to database")
    sys.exit(0)
print("Connected to database")

# Get the comments.
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Send the comments to the local filter webservice.
def do_stuff():
  session = requests.Session()
  
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  filtered = 0
  total = 0
  
  done = False
  while not done:
    cur.execute(
      """
      SELECT *
      FROM comments
      ORDER BY created ASC
      LIMIT 10000
      OFFSET %i
      """ % total
    )
    comments = cur.fetchall()
    if not comments:
      done = True
    for comment in comments:
      filterComment = { k : v for (k,v) in comment.items()
                        if k in set(['cid', 'pid', 'uid', 'comment', 'created', 'url']) }
      filter_result = session.post('http://127.0.0.1:5000/filters/filter', json=filterComment, stream=False).json()
      session.post('http://127.0.0.1:5000/filters/add_comment', json=comment, stream=False)
      
      # Some statistics.
      total += 1
      if filter_result['filtered']:
        filtered += 1
        if filter_result['result']:
          if comment['hate']:
            tp += 1
          else:
            fp += 1
        else:
          if comment['hate']:
            fn += 1
          else:
            tn += 1
      
      if total%500 == 0:
        print("Filtered %i of %i, TP: %i, FP: %i, TN: %i, FN: %i" % (filtered, total, tp, fp, tn, fn))

# cProfile.run('do_stuff()')
do_stuff()