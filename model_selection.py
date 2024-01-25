import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
class ModelSelector:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def find_best_models(self):
        X = self.df.drop(columns=['In-hospital_death', 'Survival', 'SOFA','SAPS-I'])
        y = self.df["In-hospital_death"]

        num_cols = [e for e in list(X.columns) if e not in ('Gender', 'ICUType')]

        numeric_transformer = Pipeline(steps=[
            ('num_imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_transformer, num_cols)
            ]
        )

        models_params = {
            'LR': {
                'model': LogisticRegression(),
                'params': {
                    'pca__n_components': [5, 15, 30],
                    'LR__C': [0.1, 1, 10, 100]
                }
            },
            'RF': {
                'model': RandomForestClassifier(),
                'params': {
                    'pca__n_components': [5, 15, 30],
                    'RF__n_estimators': [10, 50, 100],
                    'RF__max_depth': [5, 10, 15]
                }
            },
            'XGB': {
                'model': XGBClassifier(),
                'params': {
                    'pca__n_components': [5, 15, 30],
                    'XGB__gamma': [0.5, 1, 1.5],
                    'XGB__learning_rate': [0.01, 0.1],
                    'XGB__max_depth': [3, 5, 7],
                    'XGB__subsample': [0.8, 0.9]
                }
            },
            'NB': {
                'model': GaussianNB(),
                'params': {
                    'pca__n_components': [5, 15, 30]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'pca__n_components': [5, 15, 30],
                    'KNN__n_neighbors': [3, 5, 7]
                }
            },
            'SVM': {
                'model': SVC(),
                'params': {
                    'pca__n_components': [5, 15, 30],
                    'SVM__C': [0.1, 1, 10],
                    'SVM__kernel': ['linear', 'rbf']
                }
            },
            'ANN': {
                'model': MLPClassifier(),
                'params': {
                    'pca__n_components': [5, 15, 30],
                    'ANN__hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'ANN__activation': ['tanh', 'relu'],
                    'ANN__alpha': [0.0001, 0.001, 0.01]
                }
            }
        }

        best_models = {}
        for model_name, mp in models_params.items():
            pipeline = Pipeline(steps=[('preprocessing', preprocessor),
                                       ('pca', TruncatedSVD()),
                                       (model_name, mp['model'])])
            grid_search = GridSearchCV(pipeline, mp['params'], cv=KFold(n_splits=4), scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X, y)
            best_models[model_name] = grid_search.best_params_, grid_search.best_score_

        return best_models
# '''Best LR:
# Parameters: {'LR__C': 0.1, 'pca__n_components': 30},
# Score: 0.8244115662252739
#
# Best RF:
# Parameters: {'RF__max_depth': 10, 'RF__n_estimators': 100, 'pca__n_components': 30},
# Score: 0.8164179244473313
#
# Best XGB:
# Parameters: {'XGB__gamma': 1.5, 'XGB__learning_rate': 0.1, 'XGB__max_depth': 3, 'XGB__subsample': 0.9, 'pca__n_components': 30},
# Score: 0.8342337972790257
#
# Best NB:
# Parameters: {'pca__n_components': 5},
# Score: 0.7642612667543597
#
# Best KNN:
# Parameters: {'KNN__n_neighbors': 7, 'pca__n_components': 15},
# Score: 0.7401959227872019
#
# Best SVM:
# Parameters: {'SVM__C': 0.1, 'SVM__kernel': 'linear', 'pca__n_components': 30},
# Score: 0.8047266336719245
#
# Best ANN:
# Parameters: {'ANN__activation': 'relu', 'ANN__alpha': 0.0001, 'ANN__hidden_layer_sizes': (50,), 'pca__n_components': 15},
# Score: 0.8060660491735521'''
k = ModelSelector("Survival_dataset - Survival_dataset.csv").find_best_models()
print(k)
