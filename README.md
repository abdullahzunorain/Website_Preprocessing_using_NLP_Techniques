# Website Preprocessing Using NLP Techniques

## Overview
This project focuses on classifying websites into predefined categories based on their textual content. Utilizing Natural Language Processing (NLP) techniques, we preprocess website text data, transform it into numerical features, and train machine learning models to accurately classify websites into categories such as Travel, News, E-Commerce, and more.

## Features
- **Data Preprocessing:** Lowercasing, tokenizing, and lemmatizing website text.
- **Text Vectorization:** Converting text into numerical features using TF-IDF.
- **Model Training:** Training classification models (e.g., Naive Bayes) to categorize websites.
- **Model Evaluation:** Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.
- **Visualization:** Visualizing the distribution of website categories.

## Data
The dataset consists of website URLs, cleaned website text, and their corresponding categories. The data is preprocessed to remove unnecessary columns and encode categories into numerical labels.


## Usage
1. **Data Preprocessing:**
    - Load the dataset and preprocess the text data.
    - Convert text to lowercase, tokenize, and lemmatize the words.

2. **Text Vectorization:**
    - Use TF-IDF vectorizer to transform text data into numerical features.

3. **Model Training:**
    - Split the dataset into training and testing sets.
    - Train a classification model (e.g., Naive Bayes) on the training data.

4. **Model Evaluation:**
    - Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
    - Visualize the results.

## Example Code
```python
# Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Text Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(website_df['Website Cleaned Text']).toarray()
y = website_df['Category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```


## Future Work
- Implement advanced models like LSTM or BERT for improved performance.
- Explore additional features such as domain-specific keywords and meta tags.
- Handle imbalanced data using techniques like SMOTE.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.
