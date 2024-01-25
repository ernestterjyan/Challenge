from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class Model:
    def __init__(self):
        base_learners = [
            ('rf', RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)),
            ('xgb', XGBClassifier(gamma=0.5, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42))
        ]
        final_learner = LogisticRegression(C=0.1, random_state=42)
        self.model = StackingClassifier(estimators=base_learners, final_estimator=final_learner, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

