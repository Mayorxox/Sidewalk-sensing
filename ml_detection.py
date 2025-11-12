# ml_detection.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Example dataset (replace with your own)
data = {
    "pause_duration": [1.2, 0.4, 2.0, 0.8, 1.6, 0.3, 2.3, 1.0],
    "lateral_offset": [0.3, 0.1, 0.5, 0.2, 0.4, 0.1, 0.6, 0.3],
    "speed_change":   [0.2, 0.0, 0.3, 0.1, 0.2, 0.0, 0.4, 0.2],
    "label": ["pause", "none", "retreat", "sidestep", "pause", "none", "sidestep", "pause"]
}

df = pd.DataFrame(data)
X = df[["pause_duration", "lateral_offset", "speed_change"]]
y = df["label"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues")
plt.title("Decision Tree Yield Detection")
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png", dpi=300)
plt.show()
