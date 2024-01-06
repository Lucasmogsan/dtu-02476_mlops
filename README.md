<p align="center">
    <h1 align="center">Image Classification of Rice</h1>
    <p align="center">Repository for the final project - course <a href="https://kurser.dtu.dk/course/02476">02476</a> at DTU.</p>
</p>

# Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#rocket-usage)
- [Report](#report)
- [Project Organization](#project-organization)
- [Contributors](#wave-contributors)
- [Credits](#credits)
- [License](#license)



# Project Description

1. **Overall goal:**
The goal is to classify five different varieties of rice (Arborio, Basmati, Ipsala, Jasmine and Karacadag). A framework and model(s) is chosen alongside a simple dataset, to focus on the utilization and implementation of frameworks and tools within MLOps. The main focus will thus be on reproducability, profiling, visualizing and monitoring multiple experiments to assess model performance.

2. **Framework:**
For the project the [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) (TIMM) is being used. This is a framework/collection of models with pretrained weights, data-loaders, traning/validation scripts, and more to be used for multiple different models.

3. **Data:**
The rice image [dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data) publicly available on Kaggle contains 75.000 images with 15.000 pieces for each of the 5 classes. The total size of the dataset is 230MB. Each image is 250 x 250 pixels with a single grain of rice against a black background.

4. **Deep learning models used?**
For the classification process we will use two different Convolutional Neural Network, comparing the best and the worst network, EVA and MobileNet-V3 respectively based on the ImageNet validataion set. [(reference)](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenetv2-matched-frequency.csv)


# Installation
TBD

# :rocket: Usage
TBD


# Report
The report for the course is found in the [reports](reports/) folder.

# Project Organization
TBD

# :wave: Contributors

This project exists thanks to all the people who contribute.
<a href="https://github.com/Lucasmogsan/dtu-02476_mlops/graphs/contributors"><img src="https://opencollective.com/readme-md-generator/contributors.svg?width=890&button=false" /></a>

https://opencollective.com/dtu-02476_mlops/contributors.svg?width=890&button=false
https://github.com/Lucasmogsan/dtu-02476_mlops/graphs/contributors


# Credits
TBD

# License
TBD

