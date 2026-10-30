import sys
import torch
import soundfile as sf
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier


# ============================================================
# SPEAKER VERIFICATION SYSTEM
# ============================================================

MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

print("==============================================")
print("        SPEAKER VERIFICATION SYSTEM")
print("==============================================")
print()

print("Loading ECAPA-TDNN speaker model...")
print()

classifier = EncoderClassifier.from_hparams(
    source=MODEL_SOURCE
)

print("Speaker model loaded successfully.")
print()


# ============================================================
# LOAD AUDIO AND CREATE SPEAKER EMBEDDING
# ============================================================

def get_embedding(audio_file):

    audio, sample_rate = sf.read(
        audio_file,
        dtype="float32"
    )

    # Convert stereo audio to mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Convert NumPy array to PyTorch tensor
    signal = torch.tensor(
        audio,
        dtype=torch.float32
    )

    # Add batch dimension
    signal = signal.unsqueeze(0)

    # Convert sample rate to 16000 Hz
    if sample_rate != 16000:

        new_length = int(
            signal.shape[1] * 16000 / sample_rate
        )

        signal = F.interpolate(
            signal.unsqueeze(1),
            size=new_length,
            mode="linear",
            align_corners=False
        ).squeeze(1)

    # Generate speaker embedding
    embedding = classifier.encode_batch(
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
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Check command-line arguments
    if len(sys.argv) != 3:

        print("Usage:")
        print(
            "python src\\speaker_verify.py "
            "<reference.wav> <test.wav>"
        )

        sys.exit(1)

    reference_file = sys.argv[1]
    test_file = sys.argv[2]

    print("Reference audio:")
    print(reference_file)
    print()

    print("Test audio:")
    print(test_file)
    print()

    print("Extracting speaker embeddings...")
    print()

    # Extract embeddings
    reference_embedding = get_embedding(
        reference_file
    )

    test_embedding = get_embedding(
        test_file
    )

    # Calculate similarity
    similarity = cosine_similarity(
        reference_embedding,
        test_embedding
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("==============================================")
    print("        SPEAKER VERIFICATION RESULT")
    print("==============================================")
    print()

    print(
        "Cosine similarity:",
        round(similarity, 4)
    )

    print()

    # Temporary threshold
    threshold = 0.60

    if similarity >= threshold:

        print("Speaker verification: MATCH")

    else:

        print("Speaker verification: NO MATCH")

    print()

    print("Threshold:", threshold)

    print()
    print("==============================================")