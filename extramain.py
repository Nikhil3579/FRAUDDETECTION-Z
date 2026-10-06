@app.post(
    "/predict",
    response_model=FraudPredictionResponse
)
def predict(request: TransactionRequest):

    # --------------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------------

    if _model_pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="Fraud detection model is not loaded."
        )

    try:

        # ----------------------------------------------------------
        # CONVERT REQUEST TO DATAFRAME
        # ----------------------------------------------------------

        # Pydantic v2:
        #     model_dump()
        #
        # Pydantic v1:
        #     dict()
        #
        # Support both versions.

        if hasattr(request, "model_dump"):

            request_data = request.model_dump()

        else:

            request_data = request.dict()

        raw_df = pd.DataFrame(
            [request_data]
        )

        # ----------------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------------

        # This must use the SAME feature engineering logic
        # used during training.

        features_df = prepare_features(
            raw_df
        )

        # ----------------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------------

        probabilities = (
            _model_pipeline
            .predict_proba(features_df)
        )

        proba = float(
            probabilities[0][1]
        )

        # ----------------------------------------------------------
        # DECISION THRESHOLD
        # ----------------------------------------------------------

        is_fraud = (
            proba >= _threshold
        )

        # ----------------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------------

        risk = _risk_level(
            proba
        )

        # ----------------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------------

        return FraudPredictionResponse(

            fraud_probability=round(
                proba,
                4
            ),

            is_fraud=is_fraud,

            threshold_used=round(
                _threshold,
                4
            ),

            risk_level=risk,
        )