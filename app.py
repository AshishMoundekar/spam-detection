from flask import Flask, request, jsonify, render_template
import joblib
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
model_path = os.path.join(os.getcwd(), 'svm_model.pkl')
vectorizer_path = os.path.join(os.getcwd(), 'tfidf_vectorizer.pkl')

svm_model = joblib.load(model_path)
tfidf_vectorizer = joblib.load(vectorizer_path)

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the SMS input from the form
        input_data = request.form.get('sms')
        
        if not input_data:
            return jsonify({'error': 'No SMS provided'})

        # Preprocess input data
        input_vectorized = tfidf_vectorizer.transform([input_data])

        # Convert sparse input to dense format
        input_dense = input_vectorized.toarray()

        # Make prediction
        prediction = svm_model.predict(input_dense)[0]

        # Return the prediction
        result = {'sms': input_data, 'prediction': prediction}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)