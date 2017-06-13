from flask import *
import re

class DearEditorsFilter():
  """
  If a user directly addresses the editors its often off-topic and inappropriate.
  """
  
  # Dictionary of suspicious words.
  DICTIONARY = [ 'liebe', 'redaktion', 'zeit', 'zon' ]
  
  # How many suspicious words a comment must contain to be filtered out.
  THRESHOLD = 2
  
  def __init__(self):
    self.patterns = [
      re.compile(
        r'\b({0})\b'.format(word),
        flags=re.IGNORECASE
      )
      for word in DearEditorsFilter.DICTIONARY
    ]

  def filter(self, comment):
    text = comment['comment'].lower()
    
    count = 0
    for pattern in self.patterns:
      if pattern.search(text):
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