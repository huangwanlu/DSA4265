from flask import Flask, render_template, request, jsonify, session
from interactive_bot import interactive_chatbot
import uuid

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Required for session support
# Generate or reuse session ID
def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())[:8]
    return session["session_id"]

# Wrap chatbot call
def get_bot_response(user_input):
    session_id = get_session_id()
    return interactive_chatbot(user_input, serial_code=session_id)

# Updated: Render intro message on page load
@app.route("/")
def home():
    session_id = get_session_id()
    intro_message = interactive_chatbot("__init__", serial_code=session_id)
    return render_template("chat.html", intro_message=intro_message)

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    bot_response = get_bot_response(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
