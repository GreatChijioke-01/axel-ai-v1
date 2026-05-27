import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
import random

lemmatizer = WordNetLemmatizer()

# Load trained model and data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    """Tokenize and lemmatize input sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Create bag of words array for input sentence"""
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag, dtype=np.float32)

def predict_intent(sentence):
    """Predict intent from user input"""
    bow = bag_of_words(sentence)
    prediction = model.predict(np.array([bow]), verbose=0)[0]
    
    # Get predicted intent
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    confidence = prediction[predicted_class_index]
    
    return predicted_class, confidence

def get_response(intent):
    """Get random response for predicted intent"""
    for intent_data in intents['intents']:
        if intent_data['tag'] == intent:
            response = random.choice(intent_data['responses'])
            return response
    return "Sorry, I can't understand you."

def chat(user_input):
    """Main chat function"""
    intent, confidence = predict_intent(user_input)
    print(f"Intent: {intent} (Confidence: {confidence:.2%})")
    response = get_response(intent)
    print(f"Axel: {response}\n")

# Test the model
print("=== Axel Chatbot Tester ===\n")
print("Type 'quit' to exit\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    if not user_input:
        continue
    chat(user_input)
