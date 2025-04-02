from flask import Flask, render_template, request, jsonify
from interactive_bot import interactive_chatbot  # Import your actual chatbot logic

app = Flask(__name__)


def get_bot_response(user_input):
    return interactive_chatbot(user_input)

@app.route("/")
def home():
    return render_template("chat.html")  # Serve the HTML

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    bot_response = get_bot_response(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
