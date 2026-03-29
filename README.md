# music_genre_classification

Multi-class classification model predicting the genre of 50,000 Spotify songs across 10 genres.

## Overview
Built a full ML pipeline to classify songs into 10 genres using Spotify audio features like acousticness, danceability, energy, and tempo.

## Results
**Test AUC (OvR, Macro): 0.9022**

## Pipeline
1. Data cleaning & imputation (median for numerical, most frequent for categorical)
2. Z-score normalization + one-hot encoding
3. PCA (10 components) for dimensionality reduction
4. K-Means clustering (10 clusters) as extra features
5. Gradient Boosting Classifier

## Tools
`Python` `Scikit-learn` `Pandas` `NumPy` `Matplotlib`

## Key Finding
Classical and Hip-Hop formed the most distinct clusters in PCA space, while Rock and Hip-Hop showed significant acoustic overlap.
