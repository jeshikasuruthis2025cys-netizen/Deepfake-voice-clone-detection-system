import os
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# DEEPFAKE VOICE DETECTION
# MFCC + SVM BASELINE MODEL
# ============================================================

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"

MODEL_PATH = "models/mfcc_svm_improved.pkl"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    # Load audio at 16 kHz
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Mean of each MFCC
    mean_features = np.mean(mfcc, axis=1)

    # Standard deviation of each MFCC
    std_features = np.std(mfcc, axis=1)

    # Combine mean + standard deviation
    # 40 + 40 = 80 features
    features = np.concatenate([
        mean_features,
        std_features
    ])

    return features


# ============================================================
# LOAD DATASET
# ============================================================

X = []
y = []

print("==============================================")
print("   DEEPFAKE VOICE DETECTION - TRAINING")
print("==============================================")
print()

print("Extracting REAL audio features...")

real_count = 0

for file in sorted(os.listdir(REAL_DIR)):

    if file.lower().endswith(".wav"):

        path = os.path.join(REAL_DIR, file)

        try:
            features = extract_features(path)

            X.append(features)

            # 0 = Real
            y.append(0)

            real_count += 1

        except Exception as e:
            print("Error processing:", file)
            print(e)


print("Real samples:", real_count)

print()
print("Extracting FAKE audio features...")

fake_count = 0

for file in sorted(os.listdir(FAKE_DIR)):

    if file.lower().endswith(".wav"):

        path = os.path.join(FAKE_DIR, file)

        try:
            features = extract_features(path)

            X.append(features)

            # 1 = Fake
            y.append(1)

            fake_count += 1

        except Exception as e:
            print("Error processing:", file)
            print(e)


print("Fake samples:", fake_count)


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

X = np.array(X)
y = np.array(y)


print()
print("==============================================")
print("DATASET INFORMATION")
print("==============================================")

print("Total samples:", len(X))
print("Feature size:", X.shape[1])

print("Real samples:", np.sum(y == 0))
print("Fake samples:", np.sum(y == 1))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# CREATE SVM MODEL
# ============================================================

print()
print("==============================================")
print("TRAINING IMPROVED SVM MODEL")
print("==============================================")


model = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


print("Training completed successfully.")


# ============================================================
# TEST MODEL
# ============================================================

print()
print("Testing model...")


# Predicted class
y_pred = model.predict(X_test)


# Probability of FAKE
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


roc_auc = roc_auc_score(
    y_test,
    y_probability
)


cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("==============================================")
print("           IMPROVED MODEL RESULTS")
print("==============================================")


print()
print("Accuracy:", accuracy)


print()
print("ROC-AUC:", roc_auc)


print()
print("Confusion Matrix:")

print(cm)


print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Real",
            "Fake"
        ]
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


print()
print("==============================================")
print("MODEL SAVED SUCCESSFULLY")
print("==============================================")


print()
print("Model saved to:")

print(MODEL_PATH)


print()
print("Training completed!")