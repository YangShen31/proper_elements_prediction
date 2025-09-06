# Proper Element Prediction

### Setup
*If you want to test the pretrained models:*
First, download `merged_elements.csv` (this contains linear predictions for all asteroids) and place it in `data/`. Second, download the pre-trained `.ngb` files on Zenodo here, and place them in `data/models`

*If you want to train new models from scratch*
Download `MPCORB.DAT.gz` and `proper_catalog24.dat.gz` from [boulder.swri.edu/~davidn/Proper24](https://www2.boulder.swri.edu/~davidn/Proper24/) and place the uncompressed files in `data/`.

In a new virtual environment, install the required libraries as specified in `requirements.txt`

### Model Training
To train eccentricity and inclination models from scratch run `1_linear_prediction.py` and `2_train_models.py`

### Model Evaluation
The jupyter notebook `3_evaluate_models.ipynb` contains code to create most of the plots in the paper. Uncertainty testing and family identification code are in their respective folders as they require a bit more processing.