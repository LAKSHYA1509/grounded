FROM python:3.13-slim

WORKDIR /app

# Copy requirements first and install them in their own layer. Docker caches
# layers, so dependencies are only reinstalled when requirements.txt changes
# rather than on every source edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Read the port from the environment so one image runs anywhere: Hugging
# Face Spaces probes 7860, Render and Cloud Run inject their own $PORT.
# Hardcoding a port is what makes an image host-specific.
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
