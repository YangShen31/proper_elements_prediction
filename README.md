# Proper Element Prediction

### Setup
*If you want to test the pretrained models:*
First, download `merged_elements.csv` (this contains linear predictions for all asteroids) and place it in `data/`. Second, download the pre-trained `.ngb` files on Zenodo here, and place them in `data/models`

*If you want to train new models from scratch*
Download `MPCORB.DAT.gz` and `proper_catalog24.dat.gz` from [boulder.swri.edu/~davidn/Proper24](https://www2.boulder.swri.edu/~davidn/Proper24/) and place the uncompressed files in `data/`.

In a new virtual environment, install the required libraries as specified in `requirements.txt`

When you clone the git repo, make sure to recursively clone all submodules. This can be done with `git clone --recurse-submodules`

### Model Training
To train eccentricity and inclination models from scratch run `1_linear_prediction.py` and `2_train_models.py`

### Model Evaluation
The jupyter notebook `3_evaluate_models.ipynb` contains code to create most of the plots in the paper. Uncertainty testing and family identification code are in their respective folders as they require a bit more processing.

### Family Identification Evaluation
Download family tables from [ast.nesvorny.families_V2_0/data/families_2024](https://sbnarchive.psi.edu/pds4/non_mission/ast.nesvorny.families_V2_0/data/families_2024/). For d values thresholds, please refer to Table 3 to 7 from [Nesvorny24](https://iopscience.iop.org/article/10.3847/1538-4365/ad675c). To generate the histogram for precision and percentage detected across different families from `4_family_identification/family_identification.ipynb`, download the Table 3 to 7 and convert them into csv file. 

### Observational Uncertainty Evaluation
Download the repository from [small-body-dynamics/SBDynT](https://github.com/small-body-dynamics/SBDynT.git) to generate sample of asteroids with high observational error. 