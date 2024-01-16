# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /DTU-02476_MLOPS
#COPY ./ .
COPY .dvc/ .dvc/
COPY data.dvc data.dvc
COPY .git .git
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_group8/ mlops_group8/

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

#ENTRYPOINT ["python", "-u", "mlops_group8/container_test.py"]