from flask import Flask, render_template, request, jsonify
from model import generate_legal_argument

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form['case_details']
    argument = generate_legal_argument(user_input)
    return render_template('result.html', argument=argument)

if __name__ == '__main__':
    app.run(debug=True, port=4050)
