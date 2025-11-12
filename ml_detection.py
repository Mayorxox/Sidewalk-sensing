# ml_detection.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------
data_path = os.path.join("..", "data", "clips_annotations.csv")
df = pd.read_csv(data_path)

# Define features and label
features = ["pause_duration", "lateral_offset", "speed_change", "angle_change"]
X = df[features]
y = df["label"]

# -------------------------------------------------------------------
# Train/test split and model training
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, digits=2)
print(report)

# -------------------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap="Blues", colorbar=False)
plt.title("Decision Tree Yield Detection")
plt.tight_layout()

# Save and show
os.makedirs("../results", exist_ok=True)
plt.savefig("../results/confusion_matrix.png", dpi=300)
plt.show()
