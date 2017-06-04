import psycopg2
import psycopg2.extras
import sys
import requests
import cProfile
from filter_manager import FilterManager

# Connect to the database.
try:
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='admin'")
except:
    print("Cannot connect to database")
    sys.exit(0)
print("Connected to database")

# Get the comments.
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Classify the comments.
def do_stuff():  
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
      LIMIT 100000
      OFFSET %i
      """ % total
    )
    
    comments = cur.fetchall()
    if not comments:
      done = True
    
    for comment in comments:
      filterComment = { k : v for (k,v) in comment.items()
                        if k in set(['cid', 'pid', 'uid', 'comment', 'created', 'url']) }
      filter_result = filterManager.filter(filterComment)
      filterManager.add_comment(comment)
      
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
        print("%i,%i,%i,%i,%i,%i" % (total, filtered, tp, fp, tn, fn))

# cProfile.run('do_stuff()')
do_stuff()