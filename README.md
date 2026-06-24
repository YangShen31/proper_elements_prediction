# A machine learning model to predict proper orbital elements for main-belt asteroid family classification
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20693708.svg)](https://doi.org/10.5281/zenodo.20693708)

Arxiv release coming soon!

## Setup

In a new virtual environment (eg `python -m venv .venv`), install the required libraries as specified in `requirements.txt`. We used python 3.12, but it should work with other versions.

#### If you want to use the pretrained models:
First, download `merged_elements.csv` (this contains linear predictions for all asteroids) and place it in `data/`. Second, download the pre-trained `*.xgb` files on Zenodo [10.5281/zenodo.20693708](https://doi.org/10.5281/zenodo.20693708), and place them in `data/models`.

#### If you want to identify asteroid families:

Download family tables from [ast.nesvorny.families_V2_0/data/families_2024](https://sbnarchive.psi.edu/pds4/non_mission/ast.nesvorny.families_V2_0/data/families_2024/) into `data/family_tables` and $d$ threshold tables 3 to 7 from from [Nesvorny24](https://iopscience.iop.org/article/10.3847/1538-4365/ad675c) into `data/family_d_vals`.

#### If you want to train new models from scratch:

Download `MPCORB.DAT.gz` and `proper_catalog24.dat.gz` from [boulder.swri.edu/~davidn/Proper24](https://www2.boulder.swri.edu/~davidn/Proper24/) and place the uncompressed files in `data/`.

## Analyze New Asteroids

First, make sure you have downloaded all of the files required for using the pretrained models and identifying asteroid families.

Second, download the ASSIST ephemeris files and place them in `data/assist` as described [here](https://assist.readthedocs.io/en/latest/jupyter_examples/GettingStarted/). This is needed for the ephemeris integration we use to find the osculating elements of your body at JD2460200.5 (when the Nesvorny'24 dataset was gathered).

Now you are ready to run `full_pipeline.ipynb` which includes examples for how to predict orbital elements for asteroids in NASA horizon's database and with custom provided osculating elements. We are updating it to include an N-Body calculation of proper elements using [SBDynT](https://arxiv.org/pdf/2603.27099).

## Re-create Results from our Paper

#### Model Training

To train eccentricity and inclination models from scratch run `1_linear_prediction.py` and `2_train_models.py`

#### Model Evaluation

The jupyter notebook `3_evaluate_models.ipynb` contains code to create most of the plots in the paper. Uncertainty testing and family identification code are in their respective folders as they require a bit more processing.

#### Family Identification Evaluation

`4_family_identification/4.1_family_identification.ipynb` shows how to analyze the model performance on a single family. `4_family_identification/4.2_histogram_generations.ipynb` repeats this process for every family in the Nesvorny'24 dataset.

This code is only for *evaluating* the performance of family identification, it does not demonstrate how identify which family a newly observed asteroid is in. See [Analyze New Asteroids](#analyze-new-asteroids) and `full_pipeline.ipynb` for an example of that.
