from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['review']
    
    # Vectorize the input text
    text_vec = vectorizer.transform([text])

    # Make a prediction
    prediction = model.predict(text_vec)[0]

    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
