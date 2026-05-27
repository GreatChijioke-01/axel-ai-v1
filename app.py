from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from ddgs import DDGS

app = Flask(__name__)

# ----AXEL'S BRAIN LOADING -----
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def axel_web_search(query):
    """
    Uses DuckDUckGo to find answers when Axel is unsure.
    No API key required!
    """
    try:
        with DDGS() as ddgs:
            #WE take the first result (max_result=1)
            results = [r for r in ddgs.text(query, max_results=1, safesearch="on")]

            if results:
                # Clean up the text a bit
                snippet = results[0]['body'].replace('  ', ' ') 
                return f"{snippet} (Source: {results[0]['href']})"
            else:
                return "I searched the web, but couldn't find a clear answer for that."
    except Exception as e:
        print(f"Search Error: {e}")
        return "My web-search module is currently offline. Let's try something else!"

def predict_class(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1

    res = model.predict(np.array([bag]))[0]

    # We set the threshold to 80%. If Axel is less than 80% sure, it uses the web.
    ERROR_THRESHOLD = 0.80
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    user_msg = data.get("message").lower()
    
    # 1. NEURAL NETWORK FIRST (But only for personal/greetings)
    ints = predict_class(user_msg)
    
    if ints:
        tag = classes[ints[0][0]]
        # If it's a high-confidence personal tag, use local response
        if tag in ['greeting', 'goodbye', 'name', 'creator', 'about_sonincam', 'feedback_negative']:
            for i in intents['intents']:
                if i['tag'] == tag:
                    return jsonify({"response": random.choice(i['responses'])})

    # 2. KEYWORD SEARCH (Secondary priority)
    search_triggers = ["search", "look up", "find", "who is", "latest info", "current status"]
    # Removed "what is" from triggers so he doesn't search "what is your name"
    if any(trigger in user_msg for trigger in search_triggers):
        return jsonify({"response": axel_web_search(user_msg)})
    
    # 3. FINAL FALLBACK: If we have an intent but it's not personal, or if no intent
    if not ints:
        return jsonify({"response": axel_web_search(user_msg)})
    else:
        # Final local response for anything else we haven't caught
        tag = classes[ints[0][0]]
        for i in intents['intents']:
            if i['tag'] == tag:
                return jsonify({"response": random.choice(i['responses'])})
            
if __name__ == "__main__":
    app.run(debug=True)