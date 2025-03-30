from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

GEMINI_API_KEY = ""
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    law_prompt = """
    You are a legal expert specializing in Indian law.  
    Answer user queries based on judiciary laws, IPC (Indian Penal Code), CRPC, and legal rights.  
    Ensure your responses are factually accurate, concise, and relevant to the question.  

    Support multilingual queries and responses, including but not limited to English, Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, and Urdu.  
    Detect the user's language and provide responses accordingly, ensuring clarity and legal accuracy in the respective language.  

   
    """

    payload = {
        "contents": [{"parts": [{"text": f"{law_prompt}\n\nUser: {user_input}\n\nAI:"}]}]
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            # Format response properly for clear bullet points
            formatted_reply = (
                reply.replace("• ", "\n• ")  # Ensure bullet points are properly spaced
                     .replace(". ", ".\n")  # Add line breaks after periods for better readability
                     .replace("* **", "\n* **")  # Ensure bold sections are on new lines
            )
            
            return jsonify({"reply": formatted_reply})
        except KeyError:
            return jsonify({"error": "Invalid response format"}), 500
    else:
        return jsonify({"error": "API request failed"}), response.status_code

if __name__ == "__main__":
    app.run(debug=True, port=3000)
