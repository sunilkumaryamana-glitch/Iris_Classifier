"""Train an Iris decision tree classifier and write model + confusion matrix to outputs/."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X, y = iris.data, iris.target

    print("Features:", iris.feature_names)
    print("Target names:", iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("\nSample Predictions vs Actual:")
    for i in range(min(5, len(y_pred))):
        print(
            f"Predicted: {iris.target_names[y_pred[i]]} | "
            f"Actual: {iris.target_names[y_test[i]]}"
        )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix — Iris (test set)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    joblib.dump(model, OUTPUT_DIR / "model.joblib")
    print(f"\nSaved model to {OUTPUT_DIR / 'model.joblib'}")
    print(f"Saved confusion matrix to {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
