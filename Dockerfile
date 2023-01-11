# Stage 1 - install dependencies
FROM python:3.8-bullseye as builder

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

# In case streamlit is run inside this container
EXPOSE 8501

COPY . .
