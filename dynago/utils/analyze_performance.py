import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from itertools import cycle


def load_model_and_data():
    """Load model, test data, and label mappings"""
    clf = joblib.load("dynago/models/gesture_svm.pkl")
    df = pd.read_csv("dynago/data/test_norm.csv")

    with open("dynago/data/gesture_map.json", "r") as f:
        label_map = json.load(f)

    gesture_labels = [
        label["name"] for _, label in sorted(label_map.items(), key=lambda x: int(x[0]))
    ]

    X_test = df.drop("gesture_index", axis=1).values
    y_test = df["gesture_index"].values

    return clf, X_test, y_test, gesture_labels


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generate and display confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    _, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation="horizontal")
    plt.title("Gesture Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig("dynago/performance/confusion_matrix.png", dpi=300)
    plt.show()
    return cm


def plot_roc_curves(clf, X_test, y_test, class_names):
    """Generate ROC curves for multiclass classification"""
    y_score = clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(["blue", "red", "green", "orange", "purple", "brown"])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Gesture Classification ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig(f"dynago/performance/roc_curves.png", dpi=300)
    plt.show()

    return roc_auc


def plot_precision_recall(clf, X_test, y_test, class_names):
    """Generate Precision-Recall curves"""
    y_score = clf.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(
            y_test_bin[:, i], y_score[:, i]
        )
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    plt.figure(figsize=(10, 8))
    colors = cycle(["blue", "red", "green", "orange", "purple", "brown"])

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            label=f"{class_names[i]} (AP = {average_precision[i]:.2f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Gesture Classification Precision-Recall Curves")
    plt.savefig(f"dynago/performance/precision_recall_curves.png", dpi=300)
    plt.legend(loc="best")
    plt.show()

    return average_precision


def generate_performance_report(y_true, y_pred, class_names):
    """Print comprehensive classification report"""
    print("\n" + "=" * 60)
    print("Classification Performance Report")
    print("=" * 60 + "\n")

    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\n" + "=" * 60)
    print("Key Performance Insights")
    print("=" * 60)

    cm = confusion_matrix(y_true, y_pred)
    for i, name in enumerate(class_names):
        accuracy = cm[i, i] / cm[i, :].sum()
        misclassified = cm[i, :].sum() - cm[i, i]
        print(f"\n{name}:")
        print(f"  - Accuracy: {accuracy:.1%}")
        print(f"  - Misclassified: {misclassified} samples")
        if misclassified > 0:
            worst_class = class_names[np.argmax(cm[i, :])]
            print(f"  - Most confused with: {worst_class}")


def main():
    # Load data and model
    clf, X_test, y_test, gesture_labels = load_model_and_data()

    # Generate predictions
    y_pred = clf.predict(X_test)

    # Visualization and analysis
    # cm = plot_confusion_matrix(y_test, y_pred, gesture_labels)
    roc_auc = plot_roc_curves(clf, X_test, y_test, gesture_labels)
    avg_precision = plot_precision_recall(clf, X_test, y_test, gesture_labels)
    generate_performance_report(y_test, y_pred, gesture_labels)

    # Save metrics for comparison
    performance_metrics = {
        # "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
    }

    with open("dynago/performance/performance_metrics.json", "w") as f:
        json.dump(performance_metrics, f)


if __name__ == "__main__":
    main()
