FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential

RUN apt update -y && apt install awscli -y

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN pip install --upgrade accelerate
RUN pip uninstall -y transformers accelerate
RUN pip install transformers accelerate
RUN pip install evaluate

CMD ["python3", "app.py"]