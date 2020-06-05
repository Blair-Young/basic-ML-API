import flask
import numpy as np
from predict import Model

model = Model()

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method=='GET':
        return(flask.render_template('main.html'))
    elif flask.request.method=='POST':
        sepal_length = flask.request.form['sepal_length']
        sepal_width = flask.request.form['sepal_width']
        petal_length = flask.request.form['petal_length']
        petal_width = flask.request.form['petal_width']

        features = np.array([sepal_length, sepal_width,
                             petal_length, petal_width])
        prediction = model.predict(features=features)
        return flask.render_template('main.html', result=prediction)

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8000,debug=True)