import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Fill missing values with -1
    df.fillna(-1, inplace=True)

    def hex_to_dec(x):
        x_str = str(x).strip()
        try:
            # If it's hexadecimal
            if all(c in '0123456789abcdefABCDEF' for c in x_str):
                return int(x_str, 16)
            else:
                # Convert to float and then int if it's not hexadecimal
                return int(float(x_str))
        except ValueError:
            # Return -1 if it's an invalid value
            return -1

    # Select feature columns (all columns except 'label')
    feature_cols = [col for col in df.columns if col != 'label']

    # Apply hex_to_dec to all feature columns
    for col in feature_cols:
        df[col] = df[col].apply(hex_to_dec)

    # Split the data into features and target
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Return the processed data and train-test split
    return X_train, X_test, y_train, y_test

