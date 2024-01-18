<p align="center">
    <h1 align="center">:ear_of_rice: :rice: Image Classification of Rice :rice: :ear_of_rice: </h1>
    <h4 align="center">Repository for the final project - course <a href="https://kurser.dtu.dk/course/02476">02476</a> at DTU</h4>
</p>

<p align="center">
  <img src="assets/rice_eater.gif" alt="Animated gif rice eater" height="150">
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :book: Table of Contents

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#mortar_board-project-description"> ➤ Project Description</a></li>
    <li><a href="#hammer_and_wrench-installation"> ➤ Installation</a></li>
    <li><a href="#rocket-usage"> ➤ Usage</a></li>
    <li><a href="#computer-development"> ➤ Development</a></li>
    <li><a href="#scroll-report"> ➤ Report </a></li>
    <li><a href="#file_folder-project-organization"> ➤ Project Organization </a></li>
    <li><a href="#wave-contributors"> ➤ Contributors </a></li>
    <li><a href="#pray-credits"> ➤ Credits </a></li>
    <li><a href="#key-license"> ➤ License </a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :mortar_board: Project Description

1. **Overall goal:**
  The goal is to classify five different varieties of rice (Arborio, Basmati, Ipsala, Jasmine and Karacadag). A framework and model(s) is chosen alongside a simple dataset, to focus on the utilization and implementation of frameworks and tools within MLOps. The main focus will thus be on reproducability, profiling, visualizing and monitoring multiple experiments to assess model performance.

2. **Framework:**
  For the project the [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) (TIMM) is being used. This is a framework/collection of models with pretrained weights, data-loaders, traning/validation scripts, and more to be used for multiple different models. Only the model functionality is used in the project.

3. **Data:**
  The rice image [dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data) publicly available on Kaggle contains 75.000 images with 15.000 pieces for each of the 5 classes. The total size of the dataset is 230MB. Each image is 250 x 250 pixels with a single grain of rice against a black background. This is

4. **Deep learning models used?**
  For the classification process we will use a Convolutional Neural Network, specifically the model EVA which is the best performing model as of Jan23 based on the ImageNet validataion set [(reference)](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv).

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :hammer_and_wrench: Installation

**Local environment**

Run the following:
```bash
cd dtu-02476-mlops
make create_environment
conda activate mlops_group8
make requirements
```

**Using the cloud (GCP)**

A Google Cloud Platform (GCP) account with credits is necessary for:
- **Buckets** (storing data and model)
- **Container Registry** (storing Docker images)
- **Trigger** (automatically building the Docker images from dockerfiles from the GitHub repository)
- **Vertex AI** (running the training)
- **Cloud Run** (to host the inference API)


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :rocket: Usage

## Training
Training can be done in one of the following three ways:
1. Locally (potentially without any cloud-connection)
2. Containerized locally using the gcp buckets. The container and entrypoint is the same used in the cloud.
3. Cloud training utilizing Vertex AI as a virtual compute engine running the training image/container from the [cloudbuild_dockerfiles_train.yaml](cloudbuild_dockerfiles_train.yaml).

**1 - Locally**
1. Pull data
```bash
dvc pull -r remote_storage data.dvc # Pulls latest from gcp bucket
make data # Pulls from kaggle - this does not require gcp connection
```
2. Run training
```bash
make train-local
```
3. (if desired) Push data to gcp:
```bash
dvc add models
dvc push -r remote_storage_models_train models.dvc
```

**2 - Local container**

*NB: This uses `dvc` pull and push from/to `gcp` buckets as well as the config file specified*
See [docker/train/](docker/train/) folder for entrypoint and dockerfile.
1. Build container from dockerfile and run image:
```bash
make train-container
```
2. (If you dont want to rebuild the image) Run:
```bash
docker compose up trainer
```


**3 - In the cloud (using [Vertex AI](https://cloud.google.com/vertex-ai)):**
1. On `gcp` a `trigger` has been set up for the GitHub repository using the [cloudbuild_dockerfiles_train.yaml](cloudbuild_dockerfiles_train.yaml) every time the main branch is updated (also experimented with a webhook from the GitHub Workflows). This rebuilds the training image (from this [Dockerfile](docker/train/Dockerfile)) and thus the current config file is being used in the next step.
2. Following creates a compute instance and runs the image (pulled from gcp `container registry`). This will pull from the `data bucket`, do training, and push to the `models bucket` after training. See [docker/train/](docker/train/) folder to see entrypoint and dockerfile used.
```bash
make train-cloud
```


## Validate and test model

The validation and testing is implemented in the training loop.
- Validation evaluates the current model after each epoch during training on a set of unseen data.
- Testing uses the final model and evaluates on a new set of unseen data.

As of now it does not run independently but this could easily be implemented if desried.

## Predict
The prediction is implemented by utilizing fastapi as back-end and streamlit as frontend.

Prediction can be run in one of the following three ways
1. Locally (NB: without fastapi or streamlit)
2. Containerized (NB: you need to at least run `api-fastapi` and if frontend is desired also the `api_streamlit`)
3. Cloud Run which is activated by the trigger and [`cloudbuild_dockerfiles_api.yaml`](/cloudbuild_dockerfiles_api.yaml)

**1 - Locally**

1. Run predict

```bash
make predict_test model=<path-to-model-file> path_image=<path-to-image-file>
```

**2 - Local container**
NB: You can run both fastapi and streamlit.

1. Build container from dockerfile and run image:
```bash
make <api-fastapi/api_streamlit>
```
2. (If you dont want to rebuild the image) Run:
```bash
docker compose up <api_fastapi/api_streamlit>
```

**3 - Using API and Cloud Run**
1. On `gcp` a `trigger` has been set up for the GitHub repository using the [cloudbuild_dockerfiles_api.yaml](cloudbuild_dockerfiles_api.yaml) every time the main branch is updated. This rebuilds the api images.
2. Create a [Cloud Run](https://cloud.google.com/run?hl=en) service for each api and use the docker image in gcr.io:

Fastapi:
```bash
gcr.io/mlops-group8/api_fastapi:latest
```
Streamlit:
```bash
gcr.io/mlops-group8/api_streamlit:latest
```
The API URLs will change according to setup. For the project the [FastAPI (back-end) URL](https://run-fastapi-fz6jdlv7kq-oa.a.run.app/docs) and [Streamlit (front-end) URL](https://run-streamlit-fz6jdlv7kq-oa.a.run.app) were used (given by Cloud Run - Presumably not active anymore).

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :computer: Development
The following contains miscellaneous information useful for development.

## Data
Pull from Google Cloud Bucket (must be logged in to gcp):
```bash
dvc pull
```

Create locally from Kaggle dataset:
1. Have the `Rice_Image_Dataset` folder saved to `data/raw`

2. *Kaggle API:*
  - Go to [www.kaggle.com](www.kaggle.com) and log in
  - Go to Settings -> API -> Create new token
  - Save the json file to your local `home/user/.kaggle` folder

3. Run the make_dataset file
```bash
make data
```

To create a smaller dataset for unit tests,
```bash
make unittest_data
```

## Unit Testing
```bash
pytest tests/               # to run all unit tests
pytest tests/test_data.py   # to run a specific unit test
```

To run pytest together with coverage,
```bash
coverage run -m pytest tests/
coverage report       # to get simple coverage report
coverage report -m    # to get missing lines
coverage report -m --omit "/opt/ros/*"  # to omit certain files
```

## Pre-commit
Enable the pre-commit
```bash
pre-commit install
```
Check the commit with pre-commit
```bash
pre-commit run --all-files
```

After this you can commit as normally.
To omit/skip the pre-commit use:
```bash
git commit -m "<message>" --no-verify
```

## Timm
To see `eva` models available (use different model names if needed):
```bash
python -c "import timm; print(timm.list_models('*eva*'))"
```
Choose a model with size 224 (to match the image size in the pipeline)


## Profiling
Profiling is added to the evaluation script to show how it can be used. It can be done with the **python profilers** and **Tensorboard**.

**Using python [profilers](https://docs.python.org/3/library/profile.html)**

Saving profiling to output file:
```bash
mkdir outputs/profiling
python -m cProfile -o outputs/profiling/profiling_output.txt mlops_group8/eval_model.py
```

Show output from the file:
```bash
python mlops_group8/utility/profiling_pstats.py
```

**Using Tensorflow [Tensorboard](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)**
```bash
tensorboard --logdir=./log
```
## Google Cloud Project (GCP)
- Be aware that all services needed are enabled on gcp:

  - Cloud Build (in setting also enable Cloud Run, Service Accounts and Cloud Build)
  - Cloud Run Admin API
  - Cloud Storage (remember to make buckets public)
  - Vertex AI
  - Artifact Registry (remember to make images public)

Logs Explorer is extremely useful for logging and tracing errors on gcp.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :scroll: Report
The report for the course is found in the [reports](reports/) folder.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :file_folder: Project Organization

The directory structure of the project looks like this (minor folders and files are omitted):

```txt
├── Makefile             <- makefile with convenience commands like `make data` or `make train`
├── README.md            <- the top-level README for developers using this project.
├── data
│   ├── (processed)      <- the final data sets for modeling (only available after data pull or command)
│   ├── (raw)            <- the original, immutable data dump (only available after data pull)
│   └── test             <- test data
│
├── docker               <- dockerfiles and utilities (e.g. shell script for entrypoint)
│   ├── api_fastapi/
│   ├── api_streamlit/
│   └── train/
│
├── docker-compose.yaml  <- Docker Compose configuration file for setting up project services
│
├── docs                 <- documentation folder (NOT used)
│   ├── index.md         <- homepage for your documentation
│   ├── mkdocs.yml       <- configuration file for mkdocs
│   └── source/          <- source directory for documentation files
│
├── models               <- trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- jupyter notebooks.
│
├── pyproject.toml       <- project configuration file
│
├── reports              <- generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures          <- generated graphics and figures to be used in reporting
│   ├── README.md        <- answers to the report questions
│   └── report.py        <- script for checking the markdown file and generating a html from it
│
├── requirements.txt            <- the requirements file for reproducing the complete environment
├── requirements_dev.txt        <- the requirements file for reproducing the complete environment for developers (exteneded installtions)
├── requirements_predict.txt    <- the requirements file for reproducing the prediction environment
├── requirements_tests.txt      <- the requirements file for reproducing the test environment (unittests)
├── requirements_train.txt      <- the requirements file for reproducing the training environment
│
├── tests                <- test files for unittests
│   └── data/            <- data used for the unittests
│
├── mlops_group8         <- source code for use in this project.
│   │
│   ├── __init__.py      <- makes folder a Python module
│   │
│   ├── config           <- config files with hyperparameters and run settings
│   │   ├── __init__.py
│   │   ├── experiment/           <- individual config.yaml experiment files containing hyperparams etc.
│   │   └── default_config.yaml   <- default config file used in training referring to experiment config.yaml file
│   │
│   ├── data             <- scripts to download and/or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── utility          <- scripts used as utility functions in multiple main scripts or minor misc. scripts used for testing functions etc.
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── predict_fastapi.py    <- script for predicting from a model, hosting back-end API by fastapi
│   ├── predict_model.py      <- script for predicting from a model (used for local testing)
│   ├── streamlit_app.py      <- script for hosting front-end  APY by streamlit
│   ├── sweep_train_model.py  <- script for doing hyperparameter sweep on training the model
│   ├── train_model.py        <- script for training the model
│   └── validate_model.py     <- script for validating the model
│
├── .dvc                <- DVC configurations and cache
│   └── config          <- DVC configuration file
├── data.dvc            <- DVC file for tracking changes and versions in the data directory bucket (gcp)
├── models.dvc          <- DVC file for tracking changes and versions in the models directory bucket (gcp)
│
├── .pre-commit-config.yaml             <- dvc configurations
├── cloudbuild_dockerfiles_api.yaml     <- gcp cloudbuild file for deploying the model by the APIs
├── cloudbuild_dockerfiles_train.yaml   <- gcp cloudbuild file for building and pushing the train image
├── config_vertexai_train_cpu.yaml      <- gcp config file used for Vertex AI
│
└── LICENSE              <- open-source license if one is chosen
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :wave: Contributors
This project exists thanks to the following contributors:

<!-- readme: contributors -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/Lucasmogsan">
            <img src="https://avatars.githubusercontent.com/u/106976128?v=4" width="100;" alt="Lucasmogsan"/>
            <br />
            <sub><b>Lucas Sandby</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/yufanana">
            <img src="https://avatars.githubusercontent.com/u/58071981?v=4" width="100;" alt="yufanana"/>
            <br />
            <sub><b>Yu Fan</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/esquivelrs">
            <img src="https://avatars.githubusercontent.com/u/14069332?v=4" width="100;" alt="esquivelrs"/>
            <br />
            <sub><b>Esquivelrs</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Tran97">
            <img src="https://avatars.githubusercontent.com/u/70841724?v=4" width="100;" alt="Tran97"/>
            <br />
            <sub><b>Steven</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: contributors -end -->

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :pray: Credits

- DTU MLOps course using this [registry](https://github.com/SkafteNicki/dtu_mlops) for a great [course](https://skaftenicki.github.io/dtu_mlops/).

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :key: License
`The MIT License (MIT)`

<p align="center">
  <img src="assets/rice_meme.jpg" alt="Rice meme" height="350">
</p>

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
