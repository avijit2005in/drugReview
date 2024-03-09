FROM python:3.10

WORKDIR /app

COPY . /app

# Install core dependencies.
RUN apt-get update && apt-get install -y libpq-dev build-essential
# Install core dependencies.
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 6000

CMD ["python", "drugReviewClassification_api.py"]
