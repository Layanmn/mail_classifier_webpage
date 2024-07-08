from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the vectorizer and the pre-trained model
with open('feature_extraction.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('spam.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the message from the form
    message = request.form['message']
    
    # Transform the message using the TfidfVectorizer
    message_vectorized = tfidf_vectorizer.transform([message])
    
    # Make a prediction
    prediction = model.predict(message_vectorized)[0]
    
    # Convert numerical prediction to category
    if prediction == 1:
        category = 'Ham'
    else:
        category = 'Spam'

    return render_template('index.html', prediction_text=f'Category: {category}')

if __name__ == '__main__':
    app.run(debug=True)


