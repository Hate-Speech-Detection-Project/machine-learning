from flask import *

class ReactionFilter():
  """
  If a user posts inappropriate comments on an article, later posts by him on the same
  article are more likely to be inappropriate. This is because users are likely to complain
  about the previous decision.
  """
  
  # Size of the time window in seconds after which we forget inappropriate comments.
  WINDOW_SIZE = 600
  
  def __init__(self):
    self.users = dict()
    self.last_timestamp = -ReactionFilter.WINDOW_SIZE;
    pass

  def filter(self, comment):
    uid = comment['uid']
    created = comment['created']
    url = comment['url']
    
    # As comments are provided to filter() are sorted by their timestamp, we can just save it.
    self.last_timestamp = created
  
    if (uid, url) in self.users:
      last_inappropriate = self.users[uid, url]
      if self.__is_inside_window(last_inappropriate, created):
        return {
          'filtered': True,
          'result': True
        }
    
    return {
      'filtered': False
    }

  def add_comment(self, comment):
    uid = comment['uid']
    created = comment['created']
    url = comment['url']
    hate = comment['hate']
    
    if not hate:
      return
    
    # Only process the comment if it is inside the current sliding window.
    if self.__is_inside_window(self.last_timestamp, created):
      self.users[uid, url] = created
  
  def __is_inside_window(self, base, timestamp):
    return 0 <= timestamp - base <= ReactionFilter.WINDOW_SIZE
  
  def __clean_timestamps(self, timestamp):
    for key, last_inappropriate in self.users.items():
      if not self.__is_inside_window(last_inappropriate, timestamp):
        del self.users[key]