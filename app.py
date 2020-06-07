import os
import flask
from predict import Model
port = int(os.environ.get("PORT", 5000))


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

        features = [float(sepal_length), float(sepal_width),
                    float(petal_length), float(petal_width)]
        print(type(petal_width))
        print(f'features collected from form {features}')
        prediction = model.predict(features=features)
        return flask.render_template('main.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)

# from flask import Flask
# import os
# app = Flask(__name__)
# port = int(os.environ.get("PORT", 5000))
# @app.route('/')
# def hello_world():
#     return 'Flask Dockerized and deployed to Heroku'
# if __name__ == '__main__':
#     app.run(debug=True,host='0.0.0.0',port=port)