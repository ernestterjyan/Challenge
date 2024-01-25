import pandas as pd
import argparse
import json
import joblib
from preprocessor import Preprocessor
from model import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

class Pipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.model = Model()

    def run(self, data_path, test=False, save_model=False):
        df = pd.read_csv(data_path)

        if not test:
            X = df.drop(columns=['In-hospital_death', 'Survival', 'SOFA', 'SAPS-I'])
            y = df['In-hospital_death']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.preprocessor.fit(X_train)
            X_train_transformed = self.preprocessor.transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)

            self.model.fit(X_train_transformed, y_train)

            y_pred_proba = self.model.predict_proba(X_test_transformed)
            y_pred = (y_pred_proba > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"Accuracy: {accuracy}, ROC-AUC: {roc_auc}")

            if save_model:
                joblib.dump(self.preprocessor, 'preprocessor.pkl')
                joblib.dump(self.model, 'model.pkl')

        else:
            self.preprocessor = joblib.load('preprocessor.pkl')
            self.model = joblib.load('model.pkl')

            X = df.drop(columns=['In-hospital_death', 'Survival', 'SOFA', 'SAPS-I'])
            X_transformed = self.preprocessor.transform(X)

            predictions = self.model.predict_proba(X_transformed)
            with open('predictions.json', 'w') as file:
                json.dump({'predict_probas': predictions.tolist(), 'threshold': 0.5}, file)

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.run(args.data_path, test=args.test, save_model=args.save_model)
#Accuracy: 0.84875, ROC-AUC: 0.7626553390377196