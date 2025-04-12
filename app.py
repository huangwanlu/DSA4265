from flask import Flask, render_template, request, jsonify, session
from interactive_bot import interactive_chatbot, user_profile, chat_history  # 🔧 Import current state
import uuid
import os
import json
from interactive_bot import load_session_data_from_json

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Required for session support

# Generate or reuse session ID
def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())[:8]
    return session["session_id"]



def load_session_data(session_id):
    file_path = f"session_{session_id}.json"
    return load_session_data_from_json(file_path)


# Wrap chatbot call
def get_bot_response(user_input):
    session_id = get_session_id()
    return interactive_chatbot(user_input, serial_code=session_id)

# New welcome page route
@app.route("/")
def welcome():
    return render_template("welcome.html")

# Updated: Render intro message on page load
@app.route("/chat")
def chat():
    session_id = get_session_id()
    intro_message = interactive_chatbot("__init__", serial_code=session_id)
    return render_template("chat.html", intro_message=intro_message)

# New faq page
@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    session_id = get_session_id()

    # 🔧 Case 1: Save current session manually
    if user_input.lower() == "save chat":
        save_path = f"session_{session_id}.json"
        with open(save_path, "w") as f:
            json.dump({
                "session_id": session_id,
                "chat_history": chat_history,
                "user_profile": user_profile
            }, f, indent=2)
        return jsonify({"response": f"💾 Chat saved to `{save_path}`"})

    # 🔧 Case 2: Recover session from user-provided session ID
    if len(user_input.strip()) == 8 and user_input.strip().isalnum():
        recovered = load_session_data(user_input.strip())
        if recovered:
            session["session_id"] = user_input.strip()
            return jsonify({"response": f"✅ Session `{user_input}` loaded.\n\nYou may continue asking your question."})
        else:
            return jsonify({"response": f"⚠️ Session `{user_input}` not found."})

    # Default: normal chatbot processing
    bot_response = get_bot_response(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
