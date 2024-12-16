import pickle
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

# 1. Load the dataset
resumeDataSet = pd.read_csv('resume_dataSet.csv', encoding='utf-8')

# 2. Function to clean resumes
def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub(r'RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub(r'#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub(r'@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)  # remove non-ASCII characters
    resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

# 3. Apply cleaning function to 'Resume' column
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

# 4. Label encoding the 'Category' column
le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

# 5. Feature extraction with TfidfVectorizer
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

# 6. Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    WordFeatures, requiredTarget, random_state=42, test_size=0.2, shuffle=True, stratify=requiredTarget
)

# 7. Train the Random Forest model with OneVsRestClassifier
rf_clf = OneVsRestClassifier(RandomForestClassifier(random_state=42))
rf_clf.fit(X_train, y_train)

# 8. Save the trained model
with open('model/random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(rf_clf, model_file)

# 9. Save the Label Encoder
with open('model/label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(le, encoder_file)

# 10. Save the TfidfVectorizer
with open('model/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(word_vectorizer, vectorizer_file)

print("Trained model, label encoder, and TfidfVectorizer saved successfully!")
