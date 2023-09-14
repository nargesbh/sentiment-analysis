# Sentiment Analysis 

![logo](sentiment.png)

This repository contains a Python script for performing sentiment analysis on text data using a Naive Bayes classifier. The code is designed to preprocess and analyze sentiment in tweets from two different datasets: SandersPosNeg and OMD.

## Overview

Sentiment analysis, also known as opinion mining, is a natural language processing task that involves determining the sentiment or emotion expressed in a piece of text, such as a tweet. This code uses a Naive Bayes classifier to predict sentiment labels based on the content of tweets.

## Datasets

### 1. SandersPosNeg Dataset
- This dataset contains tweets related to a specific topic, and each tweet is labeled as either positive or negative sentiment.
- The dataset is read from a CSV file, and the preprocessing steps include removing punctuation, converting text to lowercase, tokenization, spelling correction, stop word removal, and lemmatization.
- The preprocessed text data is then transformed into numerical features using TF-IDF vectorization.
- A Naive Bayes classifier is trained on this dataset using 10-fold cross-validation, and the accuracy achieved is approximately 82.9%.

### 2. OMD (Your Dataset Name) Dataset
- This dataset contains tweets with associated sentiment labels (1 for positive, 0 for negative).
- Similar preprocessing steps are applied to this dataset, including data cleaning, tokenization, spelling correction, stop word removal, and lemmatization.
- The text data is transformed into TF-IDF vectors, and a Naive Bayes classifier is trained using 10-fold cross-validation.
- The accuracy achieved on this dataset is approximately 76.3%.

## Prerequisites

Before running the code, ensure that you have the following libraries and packages installed:

- pandas
- numpy
- string
- re
- nltk
- spellchecker
- sklearn (scikit-learn)

You can install these packages using `pip`:
```
pip install pandas numpy string nltk spellchecker scikit-learn
```

## Usage

1. Clone this repository to your local machine:

```
git clone https://github.com/your-username/sentiment-analysis.git
```

2. Navigate to the project directory:


```

cd sentiment-analysis

```

3. Run the script for sentiment analysis on your dataset:

```
python sentiment_analysis.py
```

## Results

The script provides accuracy metrics for sentiment analysis on both the SandersPosNeg dataset and the OMD dataset. You can assess the performance of the Naive Bayes classifier on your own data by following the preprocessing and analysis steps outlined in the script.


Feel free to modify and adapt this code for your own sentiment analysis tasks. If you have any questions or encounter issues, please open an issue in the GitHub repository.
