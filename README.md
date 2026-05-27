Markdown
# AXEL AI
#### Video Demo: (https://youtu.be/wLXHcIT2Uxo)
#### Description:
Axel AI is a hybrid digital assistant designed as a prototype for my future startup, **Sonincam**. 

## Features
- **Neural Network Intent Classification:** Axel uses a Keras-based model trained on a custom dataset (`intents.json`) to handle direct user interactions locally.
- **Web Fallback Search:** When the model's confidence falls below the error threshold, Axel automatically routes the query to the DuckDuckGo API to provide real-time information from the web.
- **Flask Framework:** The application is served via a Flask backend, providing a responsive interface for user interaction.
- **Threshold Intergration:** In the flask application the web fallback is activated when the confidence of Axel's response falls below a the a threshold.

## Technical Implementation
- **Data Processing:** Uses NLTK for tokenization and lemmatization of user input.
- **Model Architecture:** A deep learning sequential model with Dropout layers to prevent overfitting.
- **Environment:** Built using Python 3, TensorFlow, and Flask.

## How to Run
1. Ensure all dependencies are installed using `pip install -r requirements.txt`.
2. Run `python train.py` to train the neural network.
3. Launch the application using `flask run`.