import os
import joblib
import numpy as np
import librosa

from sklearn.metrics import accuracy_score


MODEL_PATH = "models/mfcc_svm_grouped.pkl"

REAL_DIR = "data/real"
FAKE_DIR = "data/fake"


# ---------------------------------------
# Feature extraction
# ---------------------------------------

def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40
    )

    mean_features = np.mean(
        mfcc.T,
        axis=0
    )

    std_features = np.std(
        mfcc.T,
        axis=0
    )

    return np.concatenate(
        [mean_features, std_features]
    )


# ---------------------------------------
# Add background noise
# ---------------------------------------

def add_noise(audio, noise_level=0.01):

    noise = np.random.normal(
        0,
        noise_level,
        len(audio)
    )

    noisy_audio = audio + noise

    return np.clip(
        noisy_audio,
        -1,
        1
    )


# ---------------------------------------
# Test robustness
# ---------------------------------------

print("\n========================================")
print("      ROBUSTNESS TESTING")
print("========================================")

print("\nLoading model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")


conditions = [
    ("Clean Audio", 0.00),
    ("Low Noise", 0.005),
    ("Medium Noise", 0.01),
    ("High Noise", 0.02)
]


for condition_name, noise_level in conditions:

    X = []
    y = []

    print(
        f"\nTesting: {condition_name}"
    )

    # REAL
    for filename in os.listdir(REAL_DIR):

        if not filename.endswith(".wav"):
            continue

        path = os.path.join(
            REAL_DIR,
            filename
        )

        audio, sr = librosa.load(
            path,
            sr=16000
        )

        if noise_level > 0:

            audio = add_noise(
                audio,
                noise_level
            )

        features = extract_features(
            audio
        )

        X.append(features)
        y.append(0)


    # FAKE
    for filename in os.listdir(FAKE_DIR):

        if not filename.endswith(".wav"):
            continue

        path = os.path.join(
            FAKE_DIR,
            filename
        )

        audio, sr = librosa.load(
            path,
            sr=16000
        )

        if noise_level > 0:

            audio = add_noise(
                audio,
                noise_level
            )

        features = extract_features(
            audio
        )

        X.append(features)
        y.append(1)


    X = np.array(X)
    y = np.array(y)

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )


print("\n========================================")
print("       ROBUSTNESS TEST COMPLETED")
print("========================================")