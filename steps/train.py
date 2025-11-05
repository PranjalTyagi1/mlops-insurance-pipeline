import os
import joblib
import yaml
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

class Trainer:
    def __init__(self, config_path='config.yml'):
        self.config = self.load_config(config_path)
        self.model_name = self.config['model']['name']
        self.model_params = self.config['model']['params']
        self.pipeline = self.create_pipeline()

    def load_config(self, config_path):
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def create_pipeline(self):
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
        # Drop ID and isolate target
        X = data.drop(columns=['id', 'Result'])
        y = data['Result']
        return X, y


    def train_model(self, X_train, y_train):
        print(f"🚀 Training model: {self.model_name}")
        self.pipeline.fit(X_train, y_train)
        print("✅ Model training complete")

    def save_model(self, model_out_path):
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

    # Train model
    trainer.train_model(X_train, y_train)

    # Save model
    trainer.save_model(args.model_out)


if __name__ == "__main__":
    main()
