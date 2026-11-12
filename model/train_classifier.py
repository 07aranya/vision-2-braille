import matplotlib

matplotlib.use('Agg')  # Avoid GUI backend issues on Windows

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import pickle
import os


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'data',
    'gestures.csv'
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'classifier.pkl'
)


def main():
    print("Script started")

    df = pd.read_csv(DATA_PATH)  

    print(f"Loaded {len(df)} samples across {df['label'].nunique()} labels")
    print(df['label'].value_counts())

    X = df.drop(columns=['label']).values   # no sample_id column exists — drop only 'label'
    y = df['label'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create and train KNN classifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)

    # Predict test data
    y_pred = clf.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc * 100:.2f}%")

    # Create confusion matrix
    labels = sorted(df['label'].unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(cmap='Blues')

    plt.title("Gesture Classification — Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            os.path.dirname(__file__),
            'confusion_matrix.png'
        )
    )

    print("Saved confusion matrix to model/confusion_matrix.png")

    # Train final classifier using the complete dataset
    clf_final = KNeighborsClassifier(n_neighbors=5)
    clf_final.fit(X, y)

    # Save trained model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf_final, f)

    print(f"Saved trained classifier to {MODEL_PATH}")


if __name__ == "__main__":
    main()

