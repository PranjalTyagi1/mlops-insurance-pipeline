import pandas as pd
from sklearn.impute import SimpleImputer
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_in", required=True)
    parser.add_argument("--test_in", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.test_out), exist_ok=True)

    print(f"📂 Reading train data from: {args.train_in}")
    print(f"📂 Reading test data from: {args.test_in}")

    # Load data
    train_df = pd.read_csv(args.train_in)
    test_df = pd.read_csv(args.test_in)
    # Clean currency and comma values in AnnualPremium
    for df in [train_df, test_df]:
        if 'AnnualPremium' in df.columns:
            df['AnnualPremium'] = (
                df['AnnualPremium']
                .astype(str)                      # make sure it's string
                .str.replace('[£,]', '', regex=True)  # remove £ and commas
                .astype(float)                    # convert to float
            )


    # Separate numeric and categorical columns
    numeric_cols = train_df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = train_df.select_dtypes(include=["object"]).columns

    print(f"🔢 Numeric columns: {list(numeric_cols)}")
    print(f"🔤 Categorical columns: {list(categorical_cols)}")

    # Impute numeric columns (mean)
    num_imputer = SimpleImputer(strategy="mean")
    train_num = pd.DataFrame(num_imputer.fit_transform(train_df[numeric_cols]), columns=numeric_cols)
    test_num = pd.DataFrame(num_imputer.transform(test_df[numeric_cols]), columns=numeric_cols)

    # Impute categorical columns (most frequent)
    cat_imputer = SimpleImputer(strategy="most_frequent")
    train_cat = pd.DataFrame(cat_imputer.fit_transform(train_df[categorical_cols]), columns=categorical_cols)
    test_cat = pd.DataFrame(cat_imputer.transform(test_df[categorical_cols]), columns=categorical_cols)

    # Combine back numeric + categorical
    train_clean = pd.concat([train_num, train_cat], axis=1)
    test_clean = pd.concat([test_num, test_cat], axis=1)
    

    # Save cleaned data
    train_clean.to_csv(args.train_out, index=False)
    test_clean.to_csv(args.test_out, index=False)

    print(f"✅ Cleaned train data saved to: {args.train_out}")
    print(f"✅ Cleaned test data saved to: {args.test_out}")

if __name__ == "__main__":
    main()
