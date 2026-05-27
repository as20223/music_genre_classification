# Music Genre Classification

Multi-class classification model that predicts the genre of Spotify songs across 10 genres using audio features. Built a full ML pipeline from raw data to evaluation, achieving a **macro-averaged AUC of 0.9022** on a held-out test set.

---

## The problem

Given Spotify's audio feature data — acousticness, danceability, energy, tempo, valence, and others — can a model reliably distinguish between 10 genres? The challenge is that genres like Rock and Hip-Hop share significant acoustic overlap, making clean separation non-trivial.

**Dataset:** ~50,000 songs, 10 genres, 13 audio features

---

## Pipeline

### 1. Data cleaning
Several feature columns contained `"?"` placeholders instead of nulls. Coerced all numeric features to float and replaced placeholders with `NaN` for proper imputation.

### 2. Stratified split
Manually stratified: exactly 500 test samples per genre (5,000 total), ensuring balanced evaluation across all classes regardless of dataset skew.

### 3. Preprocessing
- Numeric features: median imputation → z-score normalization
- Categorical features (`key`, `mode`): most-frequent imputation → one-hot encoding
- Built using `sklearn` `Pipeline` and `ColumnTransformer` to prevent data leakage

### 4. PCA (10 components)
Reduced the preprocessed feature space to 10 principal components, capturing the majority of variance while removing noise and reducing the dimensionality fed to clustering and classification.

### 5. K-Means cluster labels as engineered features
The most consequential design decision: ran K-Means (k=10) on the PCA-reduced training data and appended each song's cluster assignment as an additional feature for the classifier.

This works because K-Means captures non-linear genre boundaries in PCA space that the classifier would otherwise have to learn from scratch. It acts as a soft "neighborhood" signal — songs in the same cluster likely share genre-relevant acoustic properties. Classical and Hip-Hop formed the most distinct clusters; Rock and Hip-Hop showed the most overlap, consistent with the model's per-class ROC performance.

### 6. Gradient Boosting Classifier
Shallow trees (`max_depth=3`) with a low learning rate (`0.05`) and 120 estimators. Shallow depth reduces overfitting on the engineered feature space; the low learning rate allows fine-grained boosting without memorizing training noise.

---

## Results

| Metric | Score |
|---|---|
| Test AUC (OvR, Macro) | **0.9022** |
| Test samples | 5,000 (500/genre) |
| Genres | 10 |



## Key finding

Classical and Hip-Hop form the most distinct clusters in PCA space, likely because Classical has uniquely low energy/danceability and Hip-Hop has uniquely high speechiness. Rock and Hip-Hop show the most acoustic overlap — both feature high energy and loudness — which is reflected in their per-class ROC curves being the closest to the macro average.

---

## Stack

`Python` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib`
