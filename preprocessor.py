import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self):
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('pca', PCA(n_components=30, random_state=42))
        ])

    def fit(self, X):
        self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform(X)
