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
RUN pip install streamlit
RUN pip install google-cloud-storage


COPY models/ models/
COPY mlops_group8/streamlit_app.py streamlit_app.py
COPY assets/ assets/

## ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
CMD exec streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
