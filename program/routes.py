from program import app
from flask import render_template

@app.route('/')
@app.route('/index', methods=['POST'])
def index():
    if request.method == 'POST' and 'plantfeatures' in request.form:
        features = request.form.get('features')
        prediction = 1
    return render_template('/index.html', prediction=prediction)
