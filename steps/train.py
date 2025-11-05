import os
import joblib
import yaml
import argparse
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, config_path='config.yml'):
        self.config = self.load_config(config_path)
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.pipeline = self.create_pipeline()

    def load_config(self, config_path):
        """Load YAML configuration file."""
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def create_pipeline(self):
        """Build preprocessing + SMOTE + model pipeline."""
        preprocessor = ColumnTransformer(transformers=[
            ('minmax', MinMaxScaler(), ['AnnualPremium']),
            ('standardize', StandardScaler(), ['Age', 'RegionID', 'DaysSinceCreated']),
            ('onehot', OneHotEncoder(handle_unknown='ignore'),
             ['Gender', 'PastAccident', 'VehicleAge', 'HasDrivingLicense', 'SalesChannelID', 'Switch'])
        ])

        smote = SMOTE(sampling_strategy=1.0)

        model_map = {
            'RandomForestClassifier': RandomForestClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier
        }

        model_class = model_map.get(self.model_name)
        if model_class is None:
            raise ValueError(f"Unsupported model: {self.model_name}")

        model = model_class(**self.model_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])

        return pipeline

    def feature_target_separator(self, data):
        """Separate features and target column."""
        X = data.drop(columns=['id', 'Result'])
        y = data['Result']
        return X, y

    def train_model(self, X_train, y_train):
        """Train the model."""
        print(f"🚀 Training model: {self.model_name}")
        self.pipeline.fit(X_train, y_train)
        print("✅ Model training complete")

    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test data and return metrics."""
        preds = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        print(f"\n📊 Model Evaluation:")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall   : {rec:.4f}")
        print(f"   F1-score : {f1:.4f}")

        # Save metrics for DVC
        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def save_model(self, model_out_path):
        """Save trained model to specified path."""
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        joblib.dump(self.pipeline, model_out_path)
        print(f"💾 Model saved successfully at: {model_out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to cleaned training data")
    parser.add_argument("--test", required=True, help="Path to cleaned testing data")
    parser.add_argument("--model_out", required=True, help="Where to save the trained model")
    args = parser.parse_args()

    # Load data
    print(f"📂 Loading training data: {args.train}")
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    trainer = Trainer(config_path='config.yml')

    # Split features/target
    X_train, y_train = trainer.feature_target_separator(train_df)
    X_test, y_test = trainer.feature_target_separator(test_df)

    # Train and evaluate
    trainer.train_model(X_train, y_train)
    trainer.evaluate_model(X_test, y_test)

    # Save model
    trainer.save_model(args.model_out)


if __name__ == "__main__":
    main()
