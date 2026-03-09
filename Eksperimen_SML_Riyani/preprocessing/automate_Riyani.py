import pandas as pd
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop_duplicates()

    df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)

    print("Preprocessing selesai. Dataset disimpan di:", output_path)


if __name__ == "__main__":
    input_file = "dataset_raw/bank_marketing.csv"
    output_file = "dataset_preprocessing/bank_marketing_clean.csv"

    preprocess_data(input_file, output_file)