import random
random.seed(19490851)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


df = pd.read_csv("musicData.csv")
print(df.columns.tolist())

df = df.replace("?", np.nan)

for col in [
    "popularity",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

y_raw = df["music_genre"]

df = df.drop(columns=[
    "instance_id",
    "artist_name",
    "track_name",
    "obtained_date",
    "music_genre"
], errors="ignore")

train_idx = []
test_idx = []

for genre in y_raw.unique():
    genre_indices = y_raw[y_raw == genre].index.tolist()
    random.shuffle(genre_indices)
    
    test_idx.extend(genre_indices[:500])
    train_idx.extend(genre_indices[500:])

X_train = df.loc[train_idx]
X_test = df.loc[test_idx]

y_train = y_raw.loc[train_idx]
y_test = y_raw.loc[test_idx]

label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

n_classes = len(label_encoder.classes_)

numeric_features = [
    "popularity",
    "acousticness",
    "danceability",
    "duration_ms",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence"
]

categorical_features = [
    "key",
    "mode"
]

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Number of genres:", y_raw.nunique())

pca = PCA(n_components=10, random_state=19490851)
X_train_pca = pca.fit_transform(X_train_proc)
X_test_pca = pca.transform(X_test_proc)


kmeans = KMeans(n_clusters=10, random_state=19490851)
train_clusters = kmeans.fit_predict(X_train_pca)
test_clusters = kmeans.predict(X_test_pca)


X_train_final = np.hstack([X_train_pca, train_clusters.reshape(-1, 1)])
X_test_final = np.hstack([X_test_pca, test_clusters.reshape(-1, 1)])

clf = GradientBoostingClassifier(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=3,
    random_state=19490851
)

clf.fit(X_train_final, y_train_enc)

y_test_bin = label_binarize(y_test_enc, classes=range(n_classes))
y_prob = clf.predict_proba(X_test_final)

auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
print(f"\nFINAL TEST AUC (OvR, Macro): {auc:.4f}")

plt.figure(figsize=(8, 6))

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, lw=1, label=label_encoder.classes_[i])

plt.plot([0, 1], [0, 1], linestyle="--", color="black")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train_enc,
    cmap="tab10",
    alpha=0.6
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Genre Clusters in PCA Space")

plt.tight_layout()
plt.show()
