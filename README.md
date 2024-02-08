# Text Classification using Wikipedia

This project focuses on classifying texts into geographical and non-geographical categories based on the content retrieved from Wikipedia. The classification is performed using Natural Language Processing (NLP) techniques and machine learning algorithms.

# Project Overview

The goal of the project is to attribute a given English text to one of two classes: geographic or non-geographic. The implementation leverages various technologies, including NLTK (Natural Language Toolkit), Wikipedia API, and scikit-learn for machine learning.

# Project Structure

The project consists of the following components:

1. Data Retrieval:
   - Text is retrieved from Wikipedia pages using the Wikipedia API.
   - Additional text can be obtained from external websites for classification.

2. Feature Extraction:
   - Keywords extraction using NLTK for feature representation.
   - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for machine learning.

3. Classification Models:
   - Logistic Regression: A machine learning model used for binary classification.
   - Naïve Bayes: Another classification approach for comparison.

4.  Evaluation Metrics:
   - Accuracy, Precision, and Recall are calculated to assess model performance.

# Implemented Models

# Logistic Regression

The project uses logistic regression for classification, trained on a dataset of pre-annotated geographical and non-geographical Wikipedia pages.

```python
# Logistic Regression classifier
logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)
logistic_predictions = logistic_classifier.predict(X_test)
Naïve Bayes
A Naïve Bayes classifier is also implemented and evaluated against the logistic regression model.

python
Copy code
# Naïve Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)
naive_bayes_predictions = naive_bayes_classifier.predict(X_test)
Usage
To use the project, follow these steps:

Clone the repository: git clone <repository_url>
Install the required libraries: pip install -r requirements.txt
Execute the script: python your_script_name.py
Evaluation Results
Printed evaluation results for both Logistic Regression and Naïve Bayes models, including accuracy, precision, and recall.

Example
An example demonstrates how to classify the content of a website using the trained Logistic Regression model.
style_text = read_text(style_text_path)

# Generate summary and query
summary, query = generate_summary(document_text, context_window_size, style_text)
Contact
For any inquiries or assistance, please contact Mohaddeseh keshavarz at [mohaddeseh.keshavarz@studenti.univr.it].
