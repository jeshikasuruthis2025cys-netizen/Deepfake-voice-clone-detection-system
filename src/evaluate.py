import os
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


MODEL_PATH = "models/mfcc_svm_grouped.pkl"

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"


def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mean_features = np.mean(mfcc.T, axis=0)
    std_features = np.std(mfcc.T, axis=0)

    return np.concatenate(
        [mean_features, std_features]
    )


print("\n==============================================")
print("       GROUPED HELD-OUT EVALUATION")
print("==============================================")

X = []
y = []
groups = []
files = []


# REAL AUDIO
for filename in os.listdir(REAL_DIR):

    if not filename.endswith(".wav"):
        continue

    path = os.path.join(
        REAL_DIR,
        filename
    )

    X.append(
        extract_features(path)
    )

    y.append(0)

    group = filename.rsplit("_", 1)[0]

    groups.append(group)
    files.append(filename)


# FAKE AUDIO
for filename in os.listdir(FAKE_DIR):

    if not filename.endswith(".wav"):
        continue

    path = os.path.join(
        FAKE_DIR,
        filename
    )

    X.append(
        extract_features(path)
    )

    y.append(1)

    group = filename.rsplit("_", 1)[0]

    groups.append(group)
    files.append(filename)


X = np.array(X)
y = np.array(y)
groups = np.array(groups)


print(f"\nTotal samples: {len(X)}")
print(f"Feature size: {X.shape[1]}")


# -----------------------------------------
# Recreate exact grouped split
# -----------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups
    )
)


X_test = X[test_idx]
y_test = y[test_idx]


# -----------------------------------------
# Load trained model
# -----------------------------------------

print("\nLoading grouped model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded.")


# -----------------------------------------
# Test only held-out data
# -----------------------------------------

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# -----------------------------------------
# Accuracy
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities
)


# -----------------------------------------
# ROC curve
# -----------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    probabilities,
    pos_label=1
)

fnr = 1 - tpr

eer_index = np.nanargmin(
    np.abs(fpr - fnr)
)

eer = (
    fpr[eer_index] +
    fnr[eer_index]
) / 2

eer_threshold = thresholds[eer_index]


# -----------------------------------------
# Confusion matrix
# -----------------------------------------

cm = confusion_matrix(
    y_test,
    predictions
)


print("\n==============================================")
print("          FINAL HELD-OUT RESULTS")
print("==============================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC: {auc * 100:.2f}%"
)

print(
    f"EER: {eer * 100:.2f}%"
)

print(
    f"EER Threshold: {eer_threshold:.4f}"
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "REAL",
            "FAKE"
        ]
    )
)


# -----------------------------------------
# Save ROC curve
# -----------------------------------------

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "Grouped Held-Out ROC Curve"
)

plt.legend()

plt.savefig(
    "results/roc_curve_grouped.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# Save confusion matrix
# -----------------------------------------

plt.figure()

plt.imshow(cm)

plt.title(
    "Grouped Held-Out Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["REAL", "FAKE"]
)

plt.yticks(
    [0, 1],
    ["REAL", "FAKE"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.savefig(
    "results/confusion_matrix_grouped.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# Save EER results
# -----------------------------------------

with open(
    "results/eer_results_grouped.txt",
    "w"
) as file:

    file.write(
        "GROUPED HELD-OUT EVALUATION\n"
    )

    file.write(
        "===========================\n\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"ROC-AUC: {auc * 100:.2f}%\n"
    )

    file.write(
        f"EER: {eer * 100:.2f}%\n"
    )

    file.write(
        f"EER Threshold: {eer_threshold:.4f}\n"
    )

    file.write(
        f"\nConfusion Matrix:\n{cm}\n"
    )


print("\n==============================================")
print("Evaluation files saved:")
print("results/roc_curve_grouped.png")
print("results/confusion_matrix_grouped.png")
print("results/eer_results_grouped.txt")
print("==============================================")