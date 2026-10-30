import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


REAL_DIR = "data/real"
FAKE_DIR = "data/fake"

MODEL_PATH = "models/mfcc_svm_grouped.pkl"

RESULTS_DIR = "results"


def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mean_features = np.mean(mfcc, axis=1)
    std_features = np.std(mfcc, axis=1)

    return np.concatenate([
        mean_features,
        std_features
    ])


print("==============================================")
print("       CONFUSION MATRIX GENERATION")
print("==============================================")
print()

print("Loading model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print()


files = []
labels = []


print("Loading REAL audio...")

for file in sorted(os.listdir(REAL_DIR)):

    if file.lower().endswith(".wav"):

        files.append(
            os.path.join(REAL_DIR, file)
        )

        labels.append(0)


print("Loading FAKE audio...")

for file in sorted(os.listdir(FAKE_DIR)):

    if file.lower().endswith(".wav"):

        files.append(
            os.path.join(FAKE_DIR, file)
        )

        labels.append(1)


labels = np.array(labels)


print()
print("Total files:", len(files))


print()
print("Extracting features...")

X = []

for i, file in enumerate(files):

    X.append(
        extract_features(file)
    )

    if (i + 1) % 20 == 0:

        print(
            "Processed:",
            i + 1,
            "/",
            len(files)
        )


X = np.array(X)


print()
print("Making predictions...")

predictions = model.predict(X)


cm = confusion_matrix(
    labels,
    predictions
)


print()
print("==============================================")
print("CONFUSION MATRIX")
print("==============================================")

print(cm)


# Create results folder

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# Plot

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    "Deepfake Voice Detection - Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)


plt.xticks(
    [0, 1],
    ["Real", "Fake"]
)

plt.yticks(
    [0, 1],
    ["Real", "Fake"]
)


# Add numbers

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

plt.tight_layout()


output_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    output_path,
    dpi=300
)

plt.close()


print()
print("Confusion matrix saved to:")

print(output_path)

print()
print("==============================================")
print("Completed successfully!")
print("==============================================")