# Stage 1 - install dependencies
FROM python:3.7-bullseye as builder

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .
