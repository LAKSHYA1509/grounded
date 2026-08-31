FROM python:3.12-slim

WORKDIR /app

# Copy requirements first and install them in their own layer. Docker caches
# layers, so dependencies are only reinstalled when requirements.txt changes
# rather than on every source edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
