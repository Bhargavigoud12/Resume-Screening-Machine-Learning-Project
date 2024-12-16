import os
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from waitress import serve 
 


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained Random Forest model and label encoder 
model_path = r'G:\text_to_video_explanation\model\random_forest_model.pkl'
encoder_path = r'G:\text_to_video_explanation\model\label_encoder.pkl'
vectorizer_path = r'G:\text_to_video_explanation\model\tfidf_vectorizer.pkl'

with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)
    print("Model loaded successfully!")

with open(encoder_path, 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)
    print("Encoder loaded successfully!")

# Load the pre-trained TfidfVectorizer
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer loaded successfully!")

# Cleaning function
def clean_resume(resume_text):
    resume_text = re.sub(r"http\S+\s*", " ", resume_text)
    resume_text = re.sub(r"#\S+", " ", resume_text)
    resume_text = re.sub(r"@\S+", " ", resume_text)
    resume_text = re.sub(r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), " ", resume_text)
    resume_text = re.sub(r"\s+", " ", resume_text)
    return resume_text

# Route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle file upload and prediction
@app.route("/upload", methods=["POST"])
def upload_file():
    if 'resume' not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files['resume']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Read the file and clean it
        with open(file_path, 'r', encoding='utf-8') as f:
            resume_text = f.read()

        cleaned_text = clean_resume(resume_text)
        transformed_text = vectorizer.transform([cleaned_text])

        # Make a prediction
        predicted_label = rf_model.predict(transformed_text)[0]
        category = label_encoder.inverse_transform([predicted_label])[0]

        return render_template("result.html", category=category, resume_text=resume_text)

# Run the app
if __name__ == "__main__":
    serve(app, host='127.0.0.1', port=5000)

