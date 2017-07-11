from flask import Flask, jsonify, request, render_template

from predict import predict

# Init flask app
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/detect/")
def detect():
    text = request.args.get('text')
    return jsonify(predict(text))

if __name__ == '__main__':
    app.run()