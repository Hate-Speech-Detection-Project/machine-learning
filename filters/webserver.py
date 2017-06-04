from flask import *
from reaction_filter import ReactionFilter
from werkzeug.contrib.profiler import ProfilerMiddleware

class FilterManager():
  """
  Manages a number of online filters.
  """

  def __init__(self):
    # Create all the filters. Their priority is determined by their position in the list (FCFS).
    self.filters = [
      ReactionFilter()
    ]

  def filter(self, comment):
    """
    Checks if one of the filters is able to decide whether the comment is likely to
    be inappropriate or not. If 'filtered' is true, 'result' contains the decision
    (either true if the comment is likely to be okay, false if the comment is likely
    to be inappropriate).
    
    Note: Comments MUST be given to filter() in ascending order defined by their timestamp.
    """
    for filter in self.filters:
      output = filter.filter(comment)
      if output['filtered']:
        return output
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

filterManager = FilterManager()

app = Flask(__name__)

@app.route('/filters/filter', methods=["POST"])
def filter():
  comment = request.get_json()
  result = filterManager.filter(comment)
  return jsonify(result)

@app.route('/filters/add_comment', methods=["POST"])
def add_comment():
  comment = request.get_json()
  result = filterManager.add_comment(comment)
  return jsonify(result)

# app.wsgi_app = ProfilerMiddleware(app.wsgi_app)
# app.run(debug=True)
app.run()