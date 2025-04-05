from flask import Flask, render_template, request, jsonify
from interactive_bot import interactive_chatbot  # Import your actual chatbot logic

app = Flask(__name__)


def get_bot_response(user_input, session_id=None):
    return interactive_chatbot(user_input, serial_code=session_id)

@app.route("/")
def home():
    return render_template("chat.html")  # Serve the HTML

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    session_id = request.json.get("session_id")  # 👈 expect session_id from frontend
    bot_response = get_bot_response(user_input, session_id)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
