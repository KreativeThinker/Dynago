import joblib

clf = joblib.load("dynago/models/gesture_svm.pkl")

import pandas as pd

df = pd.read_csv("dynago/data/test_norm.csv")

X_test = df.drop("gesture_index", axis=1).values
y_test = df["gesture_index"].values


y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json

with open("dynago/data/gesture_map.json", "r") as f:
    label_map = json.load(f)

gesture_labels = [
    label["name"] for _, label in sorted(label_map.items(), key=lambda x: int(x[0]))
]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gesture_labels)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on Test Set")
plt.show()
