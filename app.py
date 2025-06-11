import os
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from waitress import serve 

import os

print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir())
print("Files in 'model' folder:", os.listdir('model'))


# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Instead of 'model/random_forest_model.pkl', use:
model_path      = 'random_forest_model.pkl'
encoder_path    = 'label_encoder.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'


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

# INSERT extract_text() function here 👇
from docx import Document
import PyPDF2

def extract_text(file_path, extension):
    text = ""
    if extension == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    elif extension == 'pdf':
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
    elif extension == 'docx':
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
    else:
        text = None
    return text

# Route for the homepage
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle file upload and prediction
@app.route("/uploads", methods=["POST"])
def upload_file():
    if 'resume' not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files['resume']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)

    if file:
        filename = file.filename
        extension = filename.rsplit('.', 1)[1].lower()
        allowed_extensions = {'txt', 'pdf', 'docx'}

        if extension not in allowed_extensions:
            flash("Unsupported file format. Please upload a .txt, .pdf, or .docx file.")
            return redirect(url_for('index'))

        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        resume_text = extract_text(file_path, extension)
        if not resume_text:
            flash("Failed to read the resume. Make sure it's not encrypted or damaged.")
            return redirect(url_for('index'))

        cleaned_text = clean_resume(resume_text)
        transformed_text = vectorizer.transform([cleaned_text])

        predicted_label = rf_model.predict(transformed_text)[0]
        category = label_encoder.inverse_transform([predicted_label])[0]

        return render_template("result.html", category=category, resume_text=resume_text)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


