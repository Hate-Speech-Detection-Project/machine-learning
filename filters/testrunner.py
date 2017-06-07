import psycopg2
import psycopg2.extras
import sys
import os
import errno
import requests
import cProfile
from filter_manager import FilterManager

FETCH_SIZE = 50000

def make_sure_path_exists(path):
  try:
    os.makedirs(path)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

# Connect to the database.
try:
    conn = psycopg2.connect("dbname='postgres' user='postgres' host='localhost' password='admin'")
except:
    print("Cannot connect to database")
    sys.exit(0)
print("Connected to database")

make_sure_path_exists('output')
tp_file = open('output/tp.csv', 'w')
fp_file = open('output/fp.csv', 'w')
tn_file = open('output/tn.csv', 'w')
fn_file = open('output/fn.csv', 'w')
tp_file.write('cid,pid,uid,comment,created,url,fid,timestamp,hate\n')
fp_file.write('cid,pid,uid,comment,created,url,fid,timestamp,hate\n')
tn_file.write('cid,pid,uid,comment,created,url,fid,timestamp,hate\n')
fn_file.write('cid,pid,uid,comment,created,url,fid,timestamp,hate\n')

def write_comment(file, comment):
  order = [str(comment[index]).replace('"', '""')
           for index in ('cid','pid','uid','comment','created','url','fid','timestamp','hate')]
  file.write('"' + '","'.join(order) + '"' + '\n')

# Classify the comments.
def do_stuff():  
  filterManager = FilterManager()
  
  tp = 0
  fp = 0
  tn = 0
  fn = 0
  filtered = 0
  total = 0
  
  done = False
  while not done:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
      """
      SELECT *
      FROM comments
      ORDER BY created ASC
      LIMIT %i
      OFFSET %i
      """ % (FETCH_SIZE, total)
    )
    comments = cur.fetchall()
    cur.close()
    if not comments or len(comments) < FETCH_SIZE:
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
            write_comment(tp_file, comment)
          else:
            fp += 1
            write_comment(fp_file, comment)
        else:
          if comment['hate']:
            fn += 1
            write_comment(tn_file, comment)
          else:
            tn += 1
            write_comment(fn_file, comment)
      
      if total%5000 == 0:
        print("%i,%i,%i,%i,%i,%i" % (total, filtered, tp, fp, tn, fn))
  print("%i,%i,%i,%i,%i,%i" % (total, filtered, tp, fp, tn, fn))

# cProfile.run('do_stuff()')
do_stuff()