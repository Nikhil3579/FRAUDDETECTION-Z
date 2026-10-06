FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The trained model (models/fraud_model.joblib) must exist BEFORE building
# this image, i.e. run `python -m src.train_final` locally first, or add
# a training step to your CI/CD pipeline.

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
