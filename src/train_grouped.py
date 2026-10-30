import os
import joblib
import numpy as np
import librosa

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# SETTINGS
# ============================================================

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"

MODEL_PATH = "models/mfcc_svm_grouped.pkl"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# FEATURE EXTRACTION
# ============================================================

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

    # Mean of each MFCC
    mean_features = np.mean(
        mfcc.T,
        axis=0
    )

    # Standard deviation of each MFCC
    std_features = np.std(
        mfcc.T,
        axis=0
    )

    # 40 mean + 40 standard deviation = 80 features
    features = np.concatenate(
        [mean_features, std_features]
    )

    return features


# ============================================================
# LOAD DATASET
# ============================================================

X = []
y = []
groups = []

print("\n==============================================")
print("   GROUPED DEEPFAKE VOICE DETECTION TRAINING")
print("==============================================")

print("\nLoading real audio files...")

real_files = [
    f for f in os.listdir(REAL_DIR)
    if f.lower().endswith(".wav")
]

for filename in real_files:

    file_path = os.path.join(
        REAL_DIR,
        filename
    )

    try:

        features = extract_features(file_path)

        X.append(features)

        # 0 = REAL
        y.append(0)

        # Group by original utterance ID
        group = filename.rsplit("_", 1)[0]

        groups.append(group)

    except Exception as e:

        print(
            f"Skipping {filename}: {e}"
        )


print("Loading fake audio files...")

fake_files = [
    f for f in os.listdir(FAKE_DIR)
    if f.lower().endswith(".wav")
]

for filename in fake_files:

    file_path = os.path.join(
        FAKE_DIR,
        filename
    )

    try:

        features = extract_features(file_path)

        X.append(features)

        # 1 = FAKE
        y.append(1)

        # Group by original utterance ID
        group = filename.rsplit("_", 1)[0]

        groups.append(group)

    except Exception as e:

        print(
            f"Skipping {filename}: {e}"
        )


X = np.array(X)
y = np.array(y)
groups = np.array(groups)


print("\nDataset loaded successfully.")

print(
    "Total samples:",
    len(X)
)

print(
    "Feature size:",
    X.shape[1]
)

print(
    "Real samples:",
    np.sum(y == 0)
)

print(
    "Fake samples:",
    np.sum(y == 1)
)

print(
    "Unique groups:",
    len(np.unique(groups))
)


# ============================================================
# GROUPED TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_index, test_index = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X[train_index]
X_test = X[test_index]

y_train = y[train_index]
y_test = y[test_index]

groups_train = groups[train_index]
groups_test = groups[test_index]


print("\n==============================================")
print("TRAIN / TEST SPLIT")
print("==============================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "Training groups:",
    len(np.unique(groups_train))
)

print(
    "Testing groups:",
    len(np.unique(groups_test))
)


# Check for group leakage

common_groups = set(
    groups_train
).intersection(
    set(groups_test)
)

print(
    "Common train/test groups:",
    len(common_groups)
)

if len(common_groups) == 0:

    print(
        "Group leakage check: PASSED"
    )

else:

    print(
        "WARNING: Group leakage detected!"
    )


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "svm",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=RANDOM_STATE
            )
        )
    ]
)


print("\nTraining SVM model...")

model.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)

# Probability of FAKE class (class 1)
fake_probabilities = probabilities[
    :,
    list(model.classes_).index(1)
]


# ============================================================
# EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    fake_probabilities
)

cm = confusion_matrix(
    y_test,
    y_pred
)


print("\n==============================================")
print("FINAL GROUPED EVALUATION")
print("==============================================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"ROC-AUC: {roc_auc * 100:.2f}%"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "REAL",
            "FAKE"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\n==============================================")

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print("==============================================")