import sys
import os
import numpy as np
import librosa
import joblib


MODEL_PATH = "models/mfcc_svm_improved.pkl"


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

    features = np.concatenate([
        mean_features,
        std_features
    ])

    return features


def predict_audio(file_path):

    print("Loading model...")

    model = joblib.load(MODEL_PATH)

    print("Analyzing audio...")
    print("File:", file_path)

    features = extract_features(file_path)

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    real_probability = probabilities[0]
    fake_probability = probabilities[1]

    print()
    print("==============================================")
    print("        DEEPFAKE VOICE DETECTION")
    print("==============================================")

    if prediction == 1:

        print("Prediction: FAKE")

        print(
            "Confidence: {:.2f}%".format(
                fake_probability * 100
            )
        )

    else:

        print("Prediction: REAL")

        print(
            "Confidence: {:.2f}%".format(
                real_probability * 100
            )
        )

    print()
    print("Real probability: {:.2f}%".format(
        real_probability * 100
    ))

    print("Fake probability: {:.2f}%".format(
        fake_probability * 100
    ))

    print("==============================================")


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage:")
        print(
            "python src\\predict.py <audio_file.wav>"
        )

        sys.exit(1)

    audio_file = sys.argv[1]

    if not os.path.exists(audio_file):

        print("ERROR: Audio file not found.")

        print("File:", audio_file)

        sys.exit(1)

    predict_audio(audio_file)