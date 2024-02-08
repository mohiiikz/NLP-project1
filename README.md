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

# Text Summarization with Style Transfer

This project focuses on generating abstractive summarizations of input texts following the style of another given text in the context window. The implementation is either in Python, utilizing the Natural Language Toolkit (NLTK), or in Java, using OpenNLP.

 Project Overview

The project involves the following steps:

1. Document Processing:
   - Reading and tokenizing the input document(s).
   - Computing the target lengths in a proportional way based on document length.

2. Hierarchical Summarization:
   - Slicing the document into segments within the context window.
   - Summarizing each slice with an extractive summarization approach.
   - Collating the summaries to form the final document summary.
   - Iteratively shrinking the summary until its size is within the context window.

3. Style Transfer:
   - Optionally summarizing a second document (style text) using the same approach.
   - Combining the style summary with the primary document summary.

4. Query Generation:
   - Generating a query based on the final summary.

# Project Structure

The project consists of the following components:

- `summarization.py`: The main script containing functions for document processing, hierarchical summarization, style transfer, and query generation.
- `extractive_summarization.py`: A module for extractive summarization using cosine similarity.
- `example_usage.py`: An example script demonstrating how to use the summarization functions.

# Usage

To use the project, follow these steps:

1. Clone the repository: `git clone <repository_url>`
2. Install the required libraries: `pip install -r requirements.txt`
3. Execute the script: `python example_usage.py`

# Configuration

Adjust the following parameters in the `example_usage.py` script according to your needs:

- `document_path`: Path to the input document.
- `context_window_size`: Desired context window size for summarization.

# Example

```python
# Example usage
document_path = 'path/to/your/document.txt'
context_window_size = 128
document_text = read_text(document_path)

# Optional: Provide another text for style transfer
style_text_path = 'path/to/your/style_text.txt'
style_text = read_text(style_text_path)

# Generate summary and query
summary, query = generate_summary(document_text, context_window_size, style_text)
Contact
For any inquiries or assistance, please contact Mohaddeseh keshavarz at [mohaddeseh.keshavarz@studenti.univr.it].
