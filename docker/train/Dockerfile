# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


WORKDIR /DTU-02476_MLOPS
#COPY ./ .
COPY .dvc/ .dvc/
COPY data.dvc data.dvc
COPY models.dvc models.dvc
COPY .git .git
COPY requirements_train.txt requirements.txt

#Install requirements
RUN pip install -r requirements.txt --no-cache-dir


COPY pyproject.toml pyproject.toml
COPY mlops_group8/ mlops_group8/

#Install custom package
RUN pip install . --no-deps --no-cache-dir

#Set WANDB authorization
ENV WANDB_API_KEY=54c9d96e9c632bd4175e8292c84d3b688eaa6629


# Set up the entrypoint
COPY ./docker/train/entrypoint.sh .
RUN chmod 777 / entrypoint.sh

ENTRYPOINT [ "/DTU-02476_MLOPS/entrypoint.sh" ]
