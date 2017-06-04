from flask import *

class DearEditorsFilter():
  """
  If a user directly addresses the editors its often off-topic and inappropriate.
  """
  
  # Dictionary of suspicious words.
  DICTIONARY = [ 'liebe', 'redaktion', 'zeit' ]
  
  # How many suspicious words a comment must contain to be filtered out.
  THRESHOLD = 2
  
  def __init__(self):
    pass

  def filter(self, comment):
    text = comment['comment'].lower()
    
    count = 0
    for word in DearEditorsFilter.DICTIONARY:
      if word in text:
        count += 1
        if count >= DearEditorsFilter.THRESHOLD:
          break
    else:
      return {
        'filtered': False
      }
    
    return {
      'filtered': True,
      'result': True
    }

  def add_comment(self, comment):
    pass