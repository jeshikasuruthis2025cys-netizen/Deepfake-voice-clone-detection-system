from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
import joblib

from src.features import extract_features
from src.speaker_verify import get_embedding, cosine_similarity


# ============================================================
# VOICESEC - DEEPFAKE VOICE SECURITY SYSTEM
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mfcc_svm_improved.pkl"
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

CORS(app)


# ============================================================
# LOAD SVM MODEL
# ============================================================

print("==============================================")
print("           VOICESEC BACKEND")
print("==============================================")
print()
print("Loading MFCC + SVM model...")

try:

    model = joblib.load(MODEL_PATH)

    print("SVM model loaded successfully.")

except Exception as error:

    print("ERROR: Could not load SVM model.")
    print(error)

    model = None


# ============================================================
# SPEAKER VERIFICATION THRESHOLD
# ============================================================

SPEAKER_THRESHOLD = 0.60


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "system": "VOICESEC",
        "status": "online",
        "deepfake_model": "MFCC + SVM",
        "speaker_model": "ECAPA-TDNN"
    })


# ============================================================
# VOICE ANALYSIS
# ============================================================

@app.route("/api/analyze", methods=["POST"])
def analyze_voice():

    test_file = request.files.get("testAudio")
    reference_file = request.files.get("referenceAudio")


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if test_file is None or reference_file is None:

        return jsonify({
            "success": False,
            "error": "Please upload both test voice and reference voice."
        }), 400


    if model is None:

        return jsonify({
            "success": False,
            "error": "SVM model could not be loaded."
        }), 500


    test_path = None
    reference_path = None


    try:

        print()
        print("----------------------------------------------")
        print("NEW VOICE ANALYSIS REQUEST")
        print("----------------------------------------------")

        print("Test file:")
        print(test_file.filename)

        print("Reference file:")
        print(reference_file.filename)


        # ----------------------------------------------------
        # CREATE TEMPORARY WAV FILES
        # ----------------------------------------------------

        test_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        reference_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )


        test_path = test_temp.name
        reference_path = reference_temp.name


        test_temp.close()
        reference_temp.close()


        # ----------------------------------------------------
        # SAVE UPLOADED FILES
        # ----------------------------------------------------

        test_file.save(test_path)

        reference_file.save(reference_path)


        # ====================================================
        # STAGE 1
        # MFCC FEATURE EXTRACTION
        # ====================================================

        print()
        print("Stage 1: Extracting MFCC features...")


        features = extract_features(
            test_path
        )


        print(
            "Feature vector size:",
            len(features)
        )


        # ====================================================
        # STAGE 2
        # SVM DEEPFAKE DETECTION
        # ====================================================

        print()
        print("Stage 2: Running SVM deepfake detection...")


        prediction = model.predict(
            [features]
        )[0]


        probabilities = model.predict_proba(
            [features]
        )[0]


        real_probability = float(
            probabilities[0]
        )


        fake_probability = float(
            probabilities[1]
        )


        if int(prediction) == 0:

            voice_status = "REAL"

        else:

            voice_status = "FAKE"


        print(
            "Voice status:",
            voice_status
        )

        print(
            "REAL probability:",
            round(real_probability * 100, 2),
            "%"
        )

        print(
            "FAKE probability:",
            round(fake_probability * 100, 2),
            "%"
        )


        # ====================================================
        # STAGE 3
        # ECAPA-TDNN SPEAKER VERIFICATION
        # ====================================================

        print()
        print("Stage 3: Running ECAPA-TDNN speaker verification...")


        test_embedding = get_embedding(
            test_path
        )


        reference_embedding = get_embedding(
            reference_path
        )


        similarity = cosine_similarity(
            test_embedding,
            reference_embedding
        )


        similarity = float(similarity)


        print(
            "Speaker similarity:",
            round(similarity, 4)
        )


        # ----------------------------------------------------
        # SPEAKER DECISION
        # ----------------------------------------------------

        if similarity >= SPEAKER_THRESHOLD:

            speaker_status = "MATCH"

        else:

            speaker_status = "NO MATCH"


        print(
            "Speaker status:",
            speaker_status
        )


        # ====================================================
        # STAGE 4
        # FINAL ACCESS DECISION
        # ====================================================

        print()
        print("Stage 4: Final security decision...")


        if (
            voice_status == "REAL"
            and
            speaker_status == "MATCH"
        ):

            access_status = "ACCESS GRANTED"

            security_decision = "PASSED"

        else:

            access_status = "ACCESS DENIED"

            security_decision = "BLOCKED"


        print()
        print("==============================================")
        print("FINAL RESULT")
        print("==============================================")

        print(
            "Voice Status:",
            voice_status
        )

        print(
            "Speaker Status:",
            speaker_status
        )

        print(
            "Speaker Similarity:",
            round(similarity, 4)
        )

        print(
            "Access Decision:",
            access_status
        )

        print("==============================================")


        # ====================================================
        # RETURN RESULT TO FRONTEND
        # ====================================================

        return jsonify({

            "success": True,

            "voice_status": voice_status,

            "real_probability": round(
                real_probability * 100,
                2
            ),

            "fake_probability": round(
                fake_probability * 100,
                2
            ),

            "speaker_similarity": round(
                similarity,
                4
            ),

            "speaker_status": speaker_status,

            "security_decision": security_decision,

            "access_status": access_status

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print()
        print("==============================================")
        print("ANALYSIS ERROR")
        print("==============================================")

        print(str(error))

        print("==============================================")


        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


    # ========================================================
    # CLEAN TEMPORARY FILES
    # ========================================================

    finally:

        if (
            test_path is not None
            and
            os.path.exists(test_path)
        ):

            try:

                os.remove(test_path)

            except Exception:

                pass


        if (
            reference_path is not None
            and
            os.path.exists(reference_path)
        ):

            try:

                os.remove(reference_path)

            except Exception:

                pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("          VOICESEC SYSTEM READY")
    print("==============================================")

    print()
    print("Dashboard:")
    print("http://127.0.0.1:5000")

    print()
    print("API Health:")
    print("http://127.0.0.1:5000/api/health")

    print()
    print("MFCC + SVM: READY")
    print("ECAPA-TDNN: READY")
    print("Access Control: READY")

    print()
    print("Starting Flask server...")
    print("==============================================")


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )