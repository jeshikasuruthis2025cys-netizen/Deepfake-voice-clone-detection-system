import sys
import joblib
import librosa
import numpy as np
import torch
import soundfile as sf
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/mfcc_svm_grouped.pkl"

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

SPEAKER_THRESHOLD = 0.60


# ============================================================
# MFCC FEATURE EXTRACTION
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

    mean_features = np.mean(
        mfcc.T,
        axis=0
    )

    std_features = np.std(
        mfcc.T,
        axis=0
    )

    features = np.concatenate(
        [mean_features, std_features]
    )

    return features


# ============================================================
# ECAPA SPEAKER EMBEDDING
# ============================================================

def get_embedding(
    speaker_model,
    audio_file
):

    audio, sample_rate = sf.read(
        audio_file,
        dtype="float32"
    )

    # Convert stereo to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    signal = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    # Convert to 16 kHz
    if sample_rate != 16000:

        new_length = int(
            signal.shape[1]
            * 16000
            / sample_rate
        )

        signal = F.interpolate(
            signal.unsqueeze(1),
            size=new_length,
            mode="linear",
            align_corners=False
        ).squeeze(1)

    embedding = speaker_model.encode_batch(
        signal
    )

    return embedding.squeeze()


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    embedding1,
    embedding2
):

    similarity = F.cosine_similarity(
        embedding1.unsqueeze(0),
        embedding2.unsqueeze(0)
    )

    return similarity.item()


# ============================================================
# CHECK COMMAND-LINE ARGUMENTS
# ============================================================

if len(sys.argv) != 3:

    print()
    print(
        "Usage:"
    )

    print(
        "python src\\access_control.py "
        "<test_audio> <reference_audio>"
    )

    print()

    sys.exit(1)


test_audio = sys.argv[1]

reference_audio = sys.argv[2]


# ============================================================
# HEADER
# ============================================================

print("=" * 55)

print(
    "        DEEPFAKE VOICE ACCESS CONTROL"
)

print("=" * 55)


# ============================================================
# LOAD DEEPFAKE MODEL
# ============================================================

print()

print(
    "Loading deepfake detection model..."
)

model = joblib.load(
    MODEL_PATH
)

print(
    "Deepfake model loaded successfully."
)

print(
    "Model classes:",
    model.classes_
)


# ============================================================
# LOAD SPEAKER MODEL
# ============================================================

print()

print(
    "Loading speaker verification model..."
)

speaker_model = EncoderClassifier.from_hparams(
    source=SPEAKER_MODEL
)

print(
    "Speaker model loaded successfully."
)


# ============================================================
# DISPLAY FILES
# ============================================================

print()

print(
    "Reference audio:"
)

print(
    reference_audio
)

print()

print(
    "Test audio:"
)

print(
    test_audio
)


# ============================================================
# STEP 1 — DEEPFAKE DETECTION
# ============================================================

print()

print(
    "Step 1: Detecting deepfake..."
)


# Extract EXACTLY 80 features
features = extract_features(
    test_audio
)


# Prediction
prediction = model.predict(
    [features]
)[0]


# Probability
probabilities = model.predict_proba(
    [features]
)[0]


print()

print(
    "Raw model prediction:",
    prediction
)


for class_label, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"Class {class_label} probability: "
        f"{probability:.4f}"
    )


# IMPORTANT:
# 0 = REAL
# 1 = FAKE

if int(prediction) == 0:

    voice_status = "REAL"

else:

    voice_status = "FAKE"


print()

print(
    "Voice status:",
    voice_status
)


# ============================================================
# STEP 2 — SPEAKER VERIFICATION
# ============================================================

print()

print(
    "Step 2: Verifying speaker..."
)


reference_embedding = get_embedding(
    speaker_model,
    reference_audio
)


test_embedding = get_embedding(
    speaker_model,
    test_audio
)


similarity = cosine_similarity(
    reference_embedding,
    test_embedding
)


if similarity >= SPEAKER_THRESHOLD:

    speaker_status = "MATCH"

else:

    speaker_status = "NO MATCH"


print()

print(
    f"Speaker similarity: {similarity:.4f}"
)

print(
    "Speaker verification:",
    speaker_status
)


# ============================================================
# STEP 3 — ACCESS CONTROL
# ============================================================

print()

print(
    "Step 3: Final access decision..."
)


# BOTH conditions must pass

if (
    voice_status == "REAL"
    and
    speaker_status == "MATCH"
):

    access_granted = True

else:

    access_granted = False


# ============================================================
# FINAL RESULT
# ============================================================

print()

print("=" * 55)

print(
    "             FINAL ACCESS RESULT"
)

print("=" * 55)

print(
    "Voice status:",
    voice_status
)

print(
    "Speaker status:",
    speaker_status
)

print(
    f"Speaker similarity: {similarity:.4f}"
)

print()


if access_granted:

    print(
        ">>> ACCESS GRANTED <<<"
    )

else:

    print(
        ">>> ACCESS DENIED <<<"
    )


print("=" * 55)