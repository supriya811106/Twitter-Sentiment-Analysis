# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using machine learning techniques. It includes data preprocessing, feature extraction, and classification to predict whether a tweet has a positive or negative sentiment.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

Sentiment analysis, also known as opinion mining, is the process of determining the sentiment or emotional tone expressed in a piece of text, such as a tweet. In this project, we analyze tweets from Twitter to classify them as either positive or negative sentiment.

## Data

We used a Twitter dataset containing 1.6 million tweets with labels indicating their sentiment (0 for negative and 1 for positive). The dataset includes various features, including tweet text, timestamp, and user information.

## Preprocessing

- Removed URLs from tweet text.
- Removed punctuation marks from tweet text.
- Removed stopwords (common words like "the," "and," "is") from tweet text.
- Cleaned repeated characters (e.g., "loooove" becomes "love").
- Removed numbers from tweet text.
- Removed rare words that do not contribute significantly to sentiment analysis.
- Removed special characters and extra white spaces.
- Tokenized the text and performed stemming and lemmatization.

## Feature Extraction

We used the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to convert the preprocessed text into numerical features. This technique helps in creating feature vectors for machine learning models.

## Model Training and Evaluation

We trained a Bernoulli Naive Bayes (BNB) classifier using the TF-IDF vectors as features. The model was evaluated using classification metrics and a ROC curve. The confusion matrix and ROC curve are visualized in the README for performance assessment.

## Usage

1. Clone this repository:
```bash
git clone https://github.com/supriya811106/Twitter-Sentiment-Analysis.git
```

3. Install the required dependencies (see the Dependencies section).

4. Run the Jupyter Notebook or Python script to preprocess the data, extract features, train the model, and perform sentiment analysis on new tweets.

5. Modify the code as needed for your specific use case.

## Dependencies

- Python 3.x
- Jupyter Notebook (optional, for running the project interactively)
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- NLTK (Natural Language Toolkit)
- WordCloud
- RegexpTokenizer
- TfidfVectorizer
- BernoulliNB

You can install these dependencies using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud
```
