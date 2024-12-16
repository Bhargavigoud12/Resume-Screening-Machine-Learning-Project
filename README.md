# Resume-Screening 

This is a simple AI-powered Resume Screening tool built with Flask, Python, and machine learning techniques. The tool predicts the category of a resume based on the content of the uploaded text file. It uses a Random Forest model to classify resumes into predefined categories, and it's designed for quick and efficient use.

## Features
- Upload resumes in `.txt` format.
- Categorize resumes automatically based on their content using machine learning.
- Display the predicted category along with the content of the uploaded resume.

## Tech Stack
- Backend : Flask (Python)
- Machine Learning : Random Forest Classifier (sklearn)
- Frontend : HTML, CSS
- Vectorization : TF-IDF Vectorizer
- Model & Encoding : Saved using `pickle`

## Project Structure
resume_screening/
├── app.py                  # Main Flask app file
├── save_model.py           # Script to save the trained model
├── templates/              # Folder for HTML files
│   ├── index.html          # Home page for uploading resumes
│   ├── result.html         # Page to display the result of resume classification
├── static/                 # Folder for static files (CSS, JS, images)
│   └── style.css           # CSS styles for the web pages
├── uploads/                # Folder to store uploaded resumes
├── model/                  # Folder to store the trained models and vectorizer
│   ├── random_forest_model.pkl   # Trained Random Forest model
│   ├── label_encoder.pkl        # Label encoder for categorical target values
│   └── tfidf_vectorizer.pkl     # TF-IDF vectorizer for text processing
├── resume_dataSet.csv      # Sample dataset for training the model
└── README.md               # Project description and instructions
-> execute pip install -r requirements.txt
   (Ensure requirements.txt includes necessary libraries like flask, scikit-learn, pandas, nltk, etc.)
-> run the app - python app.py

