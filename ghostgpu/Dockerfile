# GhostGPU — Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

# HF Spaces runs as a non-root user with id 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# install dependencies first so Docker can cache this layer
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# project files
COPY --chown=user src/ ./src/
COPY --chown=user static/ ./static/

USER user

# Build the dataset and train the model at image-build time, so the Space
# starts instantly instead of training on every boot.
RUN python src/generate_data.py \
 && python src/features.py \
 && python src/train_model.py

# Hugging Face Spaces expects the app on port 7860
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "src"]
