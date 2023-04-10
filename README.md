# Unsupervised hyperspectral image classification

# Project Structure
 * data/
 * models/
 * reports/
 * scripts/
 * tmp/
 * .gitignore
 * LICENSE
 * README.md

# Requirements
 * Python 3.7
 * pip
 * virtualenv

# Project configuration
`python -m venv ./venv`  
`source ./venv/bin/activate`  
`python -m pip install --upgrade pip`  
`python -m pip install -r requirements.txt`  

# Running the project

Create result directory structure and copy ground truth images to it:  
`python scripts/creator_result_files.py`

Run the project:  
`python scripts/creator_labeled_image.py`

Create spectral curves for each class:  
`python scripts/creator_mean_spectral_curve.py`

Generate report from spectral curves:  
`python scripts/analyse_spectral_curve.py`
