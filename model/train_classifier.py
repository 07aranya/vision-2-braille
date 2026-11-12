import os
import pickle
import string
import matplotlib

matplotlib.use('Agg')  # Avoid GUI backend issues on Windows
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'classifier.pkl')
CM_PATH = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')


def main():
    print("--- Vision2Braille Classifier Training ---")
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples across {df['label'].nunique()} labels")

    # Audit for all 26 letters
    all_letters = list(string.ascii_uppercase)
    present_letters = sorted(df['label'].unique().tolist())
    missing_letters = [l for l in all_letters if l not in present_letters]

    print("\nDataset Class Distribution:")
    counts = df['label'].value_counts().sort_index()
    for lbl, cnt in counts.items():
        print(f"  {lbl}: {cnt}")

    if missing_letters:
        print(f"\n[WARNING] Missing {len(missing_letters)} letters from full alphabet: {missing_letters}")
        print("  You can collect samples for these using `python data_collection/collect_gestures.py`.")
    else:
        print("\n[SUCCESS] Full 26-letter alphabet present in dataset!")

    X = df.drop(columns=['label']).values
    y = df['label'].values

    # Train / test split (80 / 20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()

    print("\n--- Test Set Evaluation ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Test Accuracy: {acc * 100:.2f}%")

    # Generate Confusion Matrix
    labels = sorted(df['label'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    plt.title(f"Vision2Braille 5-NN Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {CM_PATH}")

    # Retrain on 100% of available data for production inference
    clf_final = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    clf_final.fit(X, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf_final, f)
    print(f"Saved production classifier to {MODEL_PATH}")


if __name__ == "__main__":
    main()
