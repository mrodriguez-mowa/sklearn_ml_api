import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.model_selection import train_test_split

# PREPROCESSING 

def oversample_data_fn():
    
    df = pd.read_csv("../files/data/out_csv.csv", sep=";", encoding="latin-1")
    df = df[df["new_type"] != '2']
    df = df.drop_duplicates()

    grouped = df.groupby('new_type').size().reset_index(name='count')
    sorted_df = grouped.sort_values('count', ascending=False)

    print(len(df))

    max_count = grouped['count'].max()

    # Oversample the data
    oversampled_data = pd.DataFrame()
    for _, row in grouped.iterrows():
        label = row['new_type']
        count = row['count']
        oversampled_rows = df[df['new_type'] == label].sample(n=max_count, replace=True, random_state=42)
        oversampled_data = pd.concat([oversampled_data, oversampled_rows])

    # Reset the index of the oversampled data
    oversampled_data = oversampled_data.reset_index(drop=True)

    grouped_os = oversampled_data.groupby('new_type').size().reset_index(name='count')
    sorted_os = grouped_os.sort_values('count', ascending=False)

    print(len(oversampled_data))

    unique_labels = df['new_type'].dropna().unique()
    print(unique_labels)

    def assign_label_id(row, label_ids):
        label = row['new_type']
        return label_ids[label]

    def assign_label_ids(df, label_column):
        unique_labels = df[label_column].unique()
        label_ids = {label: i+1 for i, label in enumerate(unique_labels)}
        df['label_id'] = df.apply(lambda row: assign_label_id(row, label_ids), axis=1)
        return df

    oversampled_df = assign_label_ids(oversampled_data, 'new_type')

    oversampled_df = oversampled_data
    oversampled_df.dropna()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(oversampled_df['message'].values.astype('U'))
    y = oversampled_df['new_type']

    # Split the oversampled data into training and validation/test sets
    X_train_oversampled, X_val_oversampled, y_train_oversampled, y_val_oversampled = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train_oversampled, y_train_oversampled)

    # Evaluate the model on the oversampled validation/test data
    accuracy_oversampled = model.score(X_val_oversampled, y_val_oversampled)
    print("Oversampled Data - Accuracy: {:.2f}".format(accuracy_oversampled))

    dump(model, "../files/models/model.joblib")
    dump(vectorizer, "../files/vectorizers/vectorizer.joblib")


oversample_data_fn()