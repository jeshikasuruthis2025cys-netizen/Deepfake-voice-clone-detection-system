// ======================================================
// VOICESEC DASHBOARD
// ======================================================


// ======================================================
// PAGE LOAD
// ======================================================

document.addEventListener("DOMContentLoaded", function () {

    // --------------------------------------------------
    // SIDEBAR NAVIGATION
    // --------------------------------------------------

    const navLinks = document.querySelectorAll(".nav-link");

    navLinks.forEach(function (link) {

        link.addEventListener("click", function (event) {

            event.preventDefault();

            // Remove active class
            navLinks.forEach(function (item) {
                item.classList.remove("active");
            });

            // Add active class
            this.classList.add("active");

            // Get target section
            const targetId = this.getAttribute("href");

            const targetSection =
                document.querySelector(targetId);

            // Scroll to section
            if (targetSection) {

                targetSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

            }

        });

    });


    // --------------------------------------------------
    // TEST AUDIO FILE
    // --------------------------------------------------

    const testAudio =
        document.getElementById("testAudio");

    if (testAudio) {

        testAudio.addEventListener("change", function () {

            const info =
                document.getElementById("testFileInfo");

            if (this.files.length > 0) {

                info.textContent =
                    "Selected: " + this.files[0].name;

            } else {

                info.textContent =
                    "No file selected";

            }

        });

    }


    // --------------------------------------------------
    // REFERENCE AUDIO FILE
    // --------------------------------------------------

    const referenceAudio =
        document.getElementById("referenceAudio");

    if (referenceAudio) {

        referenceAudio.addEventListener("change", function () {

            const info =
                document.getElementById("referenceFileInfo");

            if (this.files.length > 0) {

                info.textContent =
                    "Selected: " + this.files[0].name;

            } else {

                info.textContent =
                    "No file selected";

            }

        });

    }

});


// ======================================================
// ANALYZE VOICE
// ======================================================

async function analyzeVoice() {

    const testAudio =
        document.getElementById("testAudio");

    const referenceAudio =
        document.getElementById("referenceAudio");

    const result =
        document.getElementById("result");

    const speakerResult =
        document.getElementById("speakerResult");

    const accessResult =
        document.getElementById("accessResult");

    const analyzeButton =
        document.getElementById("analyzeButton");


    // --------------------------------------------------
    // CHECK AUDIO FILES
    // --------------------------------------------------

    if (
        !testAudio ||
        !referenceAudio ||
        !testAudio.files.length ||
        !referenceAudio.files.length
    ) {

        result.innerHTML = `

            <div class="waiting">

                <div class="microphone">
                    ⚠️
                </div>

                <h3>
                    Audio files required
                </h3>

                <p>
                    Please upload both test voice
                    and reference voice.
                </p>

            </div>

        `;

        return;
    }


    // --------------------------------------------------
    // SHOW ANALYZING
    // --------------------------------------------------

    result.innerHTML = `

        <div class="waiting">

            <div class="microphone">
                🔄
            </div>

            <h3>
                Analyzing voice...
            </h3>

            <p>
                Running MFCC + SVM and
                ECAPA-TDNN verification.
            </p>

        </div>

    `;


    speakerResult.innerHTML = `

        <div class="waiting">

            <div class="microphone">
                🔄
            </div>

            <h3>
                Verifying speaker...
            </h3>

            <p>
                Comparing test voice with
                reference voice.
            </p>

        </div>

    `;


    accessResult.innerHTML = `

        <div class="waiting">

            <div class="microphone">
                🔄
            </div>

            <h3>
                Checking access...
            </h3>

            <p>
                Generating final security decision.
            </p>

        </div>

    `;


    // Disable button
    analyzeButton.disabled = true;

    analyzeButton.textContent =
        "ANALYZING...";


    // --------------------------------------------------
    // CREATE FORM DATA
    // --------------------------------------------------

    const formData = new FormData();

    formData.append(
        "testAudio",
        testAudio.files[0]
    );

    formData.append(
        "referenceAudio",
        referenceAudio.files[0]
    );


    // --------------------------------------------------
    // SEND REQUEST TO FLASK
    // --------------------------------------------------

    try {

        const response =
            await fetch("/api/analyze", {

                method: "POST",

                body: formData

            });


        // Convert response to JSON

        const data =
            await response.json();


        // --------------------------------------------------
        // CHECK BACKEND ERROR
        // --------------------------------------------------

        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                "Voice analysis failed."
            );

        }


        // --------------------------------------------------
        // DISPLAY RESULTS
        // --------------------------------------------------

        displayMainResult(data);

        displaySpeakerResult(data);

        displayAccessResult(data);

    }


    catch (error) {

        console.error(
            "Analysis error:",
            error
        );


        // Main error

        result.innerHTML = `

            <div class="waiting">

                <div class="microphone">
                    ❌
                </div>

                <h3>
                    Analysis Failed
                </h3>

                <p>
                    ${escapeHtml(error.message)}
                </p>

            </div>

        `;


        // Speaker error

        speakerResult.innerHTML = `

            <div class="waiting">

                <div class="microphone">
                    ❌
                </div>

                <h3>
                    Speaker Verification Failed
                </h3>

                <p>
                    Please check the Flask backend.
                </p>

            </div>

        `;


        // Access error

        accessResult.innerHTML = `

            <div class="waiting">

                <div class="microphone">
                    ❌
                </div>

                <h3>
                    Access Decision Unavailable
                </h3>

                <p>
                    Analysis could not be completed.
                </p>

            </div>

        `;

    }


    finally {

        // Enable button again

        analyzeButton.disabled = false;

        analyzeButton.textContent =
            "ANALYZE VOICE";

    }

}


// ======================================================
// DISPLAY MAIN VOICE RESULT
// ======================================================

function displayMainResult(data) {

    const result =
        document.getElementById("result");


    const voiceStatus =
        data.voice_status || "UNKNOWN";


    const isReal =
        voiceStatus === "REAL";


    const statusClass =
        isReal
            ? "granted"
            : "denied";


    const icon =
        isReal
            ? "✅"
            : "🚫";


    result.innerHTML = `

        <div class="result">

            <h3>
                Analysis Complete
            </h3>


            <div class="result-status ${statusClass}">

                ${icon}
                ${escapeHtml(voiceStatus)}

            </div>


            <div class="result-item">

                <span>
                    Voice Status
                </span>

                <strong>
                    ${escapeHtml(voiceStatus)}
                </strong>

            </div>


            <div class="result-item">

                <span>
                    REAL Probability
                </span>

                <strong>
                    ${data.real_probability}%
                </strong>

            </div>


            <div class="result-item">

                <span>
                    FAKE Probability
                </span>

                <strong>
                    ${data.fake_probability}%
                </strong>

            </div>


            <div class="result-item">

                <span>
                    Speaker Status
                </span>

                <strong>
                    ${escapeHtml(
                        data.speaker_status
                    )}
                </strong>

            </div>


            <div class="result-item">

                <span>
                    Speaker Similarity
                </span>

                <strong>
                    ${data.speaker_similarity}
                </strong>

            </div>


            <div class="result-item">

                <span>
                    Security Decision
                </span>

                <strong>
                    ${escapeHtml(
                        data.security_decision
                    )}
                </strong>

            </div>

        </div>

    `;

}


// ======================================================
// DISPLAY SPEAKER VERIFICATION RESULT
// ======================================================

function displaySpeakerResult(data) {

    const speakerResult =
        document.getElementById("speakerResult");


    const speakerStatus =
        data.speaker_status || "UNKNOWN";


    const similarity =
        data.speaker_similarity;


    const isMatch =
        speakerStatus === "MATCH";


    const statusClass =
        isMatch
            ? "granted"
            : "denied";


    const icon =
        isMatch
            ? "🔐"
            : "🔒";


    speakerResult.innerHTML = `

        <div class="result">

            <h3>
                Speaker Verification Complete
            </h3>


            <div class="result-status ${statusClass}">

                ${icon}
                ${escapeHtml(speakerStatus)}

            </div>


            <div class="result-item">

                <span>
                    Speaker Status
                </span>

                <strong>
                    ${escapeHtml(speakerStatus)}
                </strong>

            </div>


            <div class="result-item">

                <span>
                    Cosine Similarity
                </span>

                <strong>
                    ${similarity}
                </strong>

            </div>


            <p class="description">

                ECAPA-TDNN compares the test voice
                with the reference voice.

            </p>

        </div>

    `;

}


// ======================================================
// DISPLAY ACCESS CONTROL RESULT
// ======================================================

function displayAccessResult(data) {

    const accessResult =
        document.getElementById("accessResult");


    const accessStatus =
        data.access_status ||
        data.security_decision ||
        "ACCESS DENIED";


    const granted =
        accessStatus === "ACCESS GRANTED";


    const statusClass =
        granted
            ? "granted"
            : "denied";


    const icon =
        granted
            ? "🟢"
            : "🔴";


    accessResult.innerHTML = `

        <div class="result">

            <h3>
                Final Access Decision
            </h3>


            <div class="result-status ${statusClass}">

                ${icon}
                ${escapeHtml(accessStatus)}

            </div>


            <div class="result-item">

                <span>
                    Voice Detection
                </span>

                <strong>
                    ${escapeHtml(
                        data.voice_status
                    )}
                </strong>

            </div>


            <div class="result-item">

                <span>
                    Speaker Verification
                </span>

                <strong>
                    ${escapeHtml(
                        data.speaker_status
                    )}
                </strong>

            </div>


            <div class="result-item">

                <span>
                    Speaker Similarity
                </span>

                <strong>
                    ${data.speaker_similarity}
                </strong>

            </div>


            <div class="result-item">

                <span>
                    Security Decision
                </span>

                <strong>
                    ${escapeHtml(
                        data.security_decision
                    )}
                </strong>

            </div>

        </div>

    `;

}


// ======================================================
// HTML ESCAPE
// ======================================================

function escapeHtml(value) {

    if (
        value === null ||
        value === undefined
    ) {

        return "";

    }


    return String(value)

        .replace(/&/g, "&amp;")

        .replace(/</g, "&lt;")

        .replace(/>/g, "&gt;")

        .replace(/"/g, "&quot;")

        .replace(/'/g, "&#039;");

}