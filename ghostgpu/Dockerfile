# GhostGPU — container image (works on Render, Hugging Face Spaces, etc.)
FROM python:3.11-slim

# create a non-root user and make /app owned by that user
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown -R user:user /app

# install dependencies first (cached layer)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# project files
COPY --chown=user:user src/ ./src/
COPY --chown=user:user static/ ./static/

# switch to the non-root user AFTER /app is fully owned by it
USER user

# build the dataset + train the model at image-build time so the app
# starts instantly instead of training on every boot
RUN python src/generate_data.py \
 && python src/features.py \
 && python src/train_model.py

# the platform sets $PORT (Render); default to 7860 (HF Spaces) if unset
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --app-dir src"]
