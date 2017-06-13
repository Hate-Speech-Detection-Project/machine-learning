from reaction_filter import ReactionFilter
from dear_editors_filter import DearEditorsFilter

class FilterManager():
  """
  Manages a number of online filters.
  """

  def __init__(self):
    # Create all the filters. Their priority is determined by their position in the list (FCFS).
    self.filters = [
      ReactionFilter(600),
      DearEditorsFilter()
    ]

  def filter(self, comment):
    """
    Checks if one of the filters is able to decide whether the comment is likely to
    be inappropriate or not. If 'filtered' is true, 'result' contains the decision
    (either true if the comment is likely to be okay, false if the comment is likely
    to be inappropriate). If the filter is not sure, 'filtered' will be false.
    
    Note: Comments MUST be given to filter() in ascending order defined by their timestamp.
    """
    result = True
    for filter in self.filters:
      output = filter.filter(comment)
      if not output['filtered'] or not output['result']:
        result = False
        break
    if result:
      return {
        'filtered': True,
        'result': True
      }
    return {
      'filtered': False
    }

  def add_comment(self, comment):
    """
    Adds a tagged comment to the filters' ground truth. This is separate from the filter()
    method because moderators usually need some time after the posting of a comment to
    actually evaluate it.
    """
    for filter in self.filters:
      filter.add_comment(comment)