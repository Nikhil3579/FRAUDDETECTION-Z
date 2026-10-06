// ============================================================
// Fraud Detection Dashboard
// ============================================================

// IMPORTANT:
// Local API
const API_URL = "https://frauddetection-z.onrender.com";

// After Render deployment, change it to:
//
// const API_URL = "https://YOUR-API.onrender.com";


// ============================================================
// SET TODAY'S DATE
// ============================================================

document.getElementById("Date").value =
    new Date().toISOString().split("T")[0];


// ============================================================
// GET INPUT VALUE
// ============================================================

function getValue(id) {

    return document
        .getElementById(id)
        .value;
}


// ============================================================
// PREDICT FRAUD
// ============================================================

async function predictFraud() {

    const button =
        document.getElementById("predictButton");

    const result =
        document.getElementById("result");


    // --------------------------------------------------------
    // Loading
    // --------------------------------------------------------

    button.disabled = true;

    button.innerText = "ANALYZING TRANSACTION...";


    result.innerHTML = `
        <div class="result-box">
            <div class="result-title">
                ANALYZING
            </div>

            <p>
                Sending transaction to AI model...
            </p>
        </div>
    `;


    // --------------------------------------------------------
    // Request body
    // --------------------------------------------------------

    const transaction = {

        Transaction_Amount:
            Number(
                getValue("Transaction_Amount")
            ),

        Account_Balance:
            Number(
                getValue("Account_Balance")
            ),

        Previous_Fraudulent_Activity:
            Number(
                getValue(
                    "Previous_Fraudulent_Activity"
                )
            ),

        Daily_Transaction_Count:
            Number(
                getValue(
                    "Daily_Transaction_Count"
                )
            ),

        Card_Age:
            Number(
                getValue("Card_Age")
            ),

        Transaction_Type:
            getValue("Transaction_Type"),

        Device_Type:
            getValue("Device_Type"),

        Location:
            getValue("Location"),

        Merchant_Category:
            getValue("Merchant_Category"),

        Card_Type:
            getValue("Card_Type"),

        Date:
            getValue("Date")
    };


    try {

        // ----------------------------------------------------
        // API CALL
        // ----------------------------------------------------

        const response =
            await fetch(
                `${API_URL}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            transaction
                        )
                }
            );


        if (!response.ok) {

            const error =
                await response.json();

            throw new Error(
                error.detail ||
                "Prediction failed"
            );
        }


        // ----------------------------------------------------
        // API RESPONSE
        // ----------------------------------------------------

        const data =
            await response.json();


        // ----------------------------------------------------
        // DISPLAY RESULT
        // ----------------------------------------------------

        displayResult(data);


    } catch (error) {

        result.innerHTML = `

            <div class="result-box">

                <div class="result-title">
                    API ERROR
                </div>

                <p>
                    ${error.message}
                </p>

            </div>

        `;

        console.error(error);

    } finally {

        button.disabled = false;

        button.innerText =
            "CHECK TRANSACTION";
    }
}


// ============================================================
// DISPLAY RESULT
// ============================================================

function displayResult(data) {

    const result =
        document.getElementById("result");


    const probability =
        (
            data.fraud_probability * 100
        ).toFixed(2);


    const threshold =
        (
            data.threshold_used * 100
        ).toFixed(2);


    let title;

    if (data.is_fraud) {

        title = "⚠ FRAUD DETECTED";

    } else {

        title = "✓ TRANSACTION SAFE";
    }


    result.innerHTML = `

        <div class="result-box">

            <div class="result-title">

                ${title}

            </div>


            <div class="probability">

                ${probability}%

            </div>


            <p>
                Fraud Probability
            </p>


            <div class="result-info">

                <div class="info">

                    <div class="info-label">
                        DECISION THRESHOLD
                    </div>

                    <div class="info-value">
                        ${threshold}%
                    </div>

                </div>


                <div class="info">

                    <div class="info-label">
                        RISK LEVEL
                    </div>

                    <div class="info-value">
                        ${data.risk_level}
                    </div>

                </div>

            </div>

        </div>

    `;
}
