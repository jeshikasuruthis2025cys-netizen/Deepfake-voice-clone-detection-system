import librosa
import numpy as np


def extract_features(file_path):

    # Load audio at 16 kHz
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000
    )

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Calculate mean for each MFCC
    mean_features = np.mean(
        mfcc.T,
        axis=0
    )

    # Calculate standard deviation for each MFCC
    std_features = np.std(
        mfcc.T,
        axis=0
    )

    # Combine:
    # 40 mean features + 40 standard deviation features
    # = 80 features
    features = np.concatenate([
        mean_features,
        std_features
    ])

    return features


# Test the feature extraction module
if __name__ == "__main__":

    print("Feature extraction module is ready.")