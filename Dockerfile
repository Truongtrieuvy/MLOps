FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# copy code + model
COPY app/ /app/

EXPOSE 5000
CMD ["python", "app.py"]
