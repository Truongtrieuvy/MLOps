FROM python:3.10-slim

WORKDIR /app

# copy file yêu cầu thư viện
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy toàn bộ source
COPY . .

EXPOSE 5000

CMD ["python", "app/app.py"]
