from flask import Flask, request, render_template
from chatbot import get_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response_route():
    user_input = request.form['user_input']
    bot_response = get_response(user_input)
    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
