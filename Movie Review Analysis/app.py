from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and preprocessor
with open('model.pkl', 'rb') as model_file:
    baseline_model = pickle.load(model_file)

with open('preprocessor.pkl', 'rb') as preprocessor_file:
    preprocessor = pickle.load(preprocessor_file)

# Define feature columns
features = ['reviewText', 'originalLanguage', 'genre', 'runtimeMinutes', 'audienceScore', 'rating', 'ratingContents', 'distributor', 'director']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['moviereview']
    
    # Create a DataFrame for the input
    input_data = pd.DataFrame(columns=features)
    input_data.at[0, 'reviewText'] = review_text
    
    # Default values for other features
    input_data.at[0, 'originalLanguage'] = 'en'  # Default language
    input_data.at[0, 'genre'] = 'Drama'  # Default genre
    input_data.at[0, 'runtimeMinutes'] = 120  # Default runtime
    input_data.at[0, 'audienceScore'] = 70  # Default audience score
    input_data.at[0, 'rating'] = 'PG-13'  # Default rating
    input_data.at[0, 'ratingContents'] = 'Mild'  # Default rating content
    input_data.at[0, 'distributor'] = 'Universal'  # Default distributor
    input_data.at[0, 'director'] = 'John Doe'  # Default director
    
    # Apply preprocessor
    processed_data = preprocessor.transform(input_data)
    
    # Predict sentiment
    prediction = baseline_model.predict(processed_data)
    
    return render_template('index.html', prediction_text=f'Review Sentiment: {prediction[0]}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
