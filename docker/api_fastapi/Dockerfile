FROM python:3.11-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_predict.txt requirements.txt
COPY mlops_group8/ mlops_group8/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install python-multipart
RUN pip install google-cloud-storage


COPY models/ models/
COPY mlops_group8/predict_fastapi.py predict_fastapi.py

ENV PORT=8000

##CMD exec predict_fastapi:app --port $PORT --host 0.0.0.0 --workers 1
CMD exec uvicorn predict_fastapi:app --port $PORT --host 0.0.0.0 --workers 1
