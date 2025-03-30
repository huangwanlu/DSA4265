from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Chatbot logic (replace with your actual AI/API calls)
def get_bot_response(user_input):
    responses = [
        f"I understand you're asking about: {user_input}",
        f"That's an interesting point about {user_input}",
        f"Let me think about {user_input}",
        f"I'm analyzing your question: {user_input}",
        f"Thanks for sharing: {user_input}"
    ]
    return responses[0]  # Or use random.choice(responses)

@app.route("/")
def home():
    return render_template("chat_new.html")  # Serve the HTML

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    bot_response = get_bot_response(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)