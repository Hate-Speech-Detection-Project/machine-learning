from flask import *
from filter_manager import FilterManager
from werkzeug.contrib.profiler import ProfilerMiddleware

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