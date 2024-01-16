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

**Locally**
1. Pull data
```bash
dvc pull -r remote_storage data.dvc
```
2. Run training
```bash
make train
```
3. (if desired) Push data:
```bash
dvc add models
dvc push -r remote_storage_models_train models.dvc
```

**Local container**

*NB: This uses `dvc` pull and push from/to `gcp` buckets.*
See [docker/train/](docker/train/) folder for entrypoint and dockerfile.
1. Build container from dockerfile and run image:
```bash
make train-container
```
2. (If you dont want to rebuild the image) Run:
```bash
docker compose up trainer
```


**In the cloud (using [Vertex AI](https://cloud.google.com/vertex-ai)):**
1. On `gcp` a `trigger` has been set up for the GitHub repository using the [cloudbuild_dockerfiles.yaml](cloudbuild_dockerfiles.yaml) every time the main branch is updated. This rebuilds the training image (from this [Dockerfile](docker/train/Dockerfile)).
2. Following creates a compute instance and runs the image (pulled from gcp `container registry`). This will pull from the `data bucket`, do training, and push to the `models bucket` after training. See [docker/train/](docker/train/) folder to see entrypoint and dockerfile used.
```bash
make train-cloud
```


## Validate

**Locally**

NB: You need a model in `models` folder and specify this in your `config` file.

```bash
make validate
```

## Predict

**Locally**

1. Run predict

```bash
make predict_test model=<path-to-model-file> path_image=<path-to-image-file>
```

**Local container**
NB: You can run both fastapi and streamlit.

1. Build container from dockerfile and run image:
```bash
make <api-fastapi/api_streamlit>
```
2. (If you dont want to rebuild the image) Run:
```bash
docker compose up <api_fastapi/api_streamlit>
```

**Using API and Cloud Run**

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


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :computer: Development


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


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :scroll: Report
The report for the course is found in the [reports](reports/) folder.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

# :file_folder: Project Organization

The directory structure of the project looks like this:

```txt
├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── mlops_group8  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
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
TBD

<p align="center">
  <img src="assets/rice_meme.jpg" alt="Rice meme" height="350">
</p>

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).
