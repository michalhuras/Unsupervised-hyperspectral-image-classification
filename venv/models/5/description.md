# Description


# Dataset:
jasperRidge2_R198.mat

# Autoencoder: 

Autoencoder(
  (encoder): Sequential(
    (0): Linear(in_features=200, out_features=90, bias=True)
    (1): Linear(in_features=90, out_features=40, bias=True)
    (2): Linear(in_features=40, out_features=20, bias=True)
  )
  (decoder): Sequential(
    (0): Linear(in_features=20, out_features=40, bias=True)
    (1): Linear(in_features=40, out_features=90, bias=True)
    (2): Linear(in_features=90, out_features=200, bias=True)
  )
)

# Clustering

Mean-Shift Clustering

# Notes

Number of clasters for jasperRidge2_R198 dataset: 4