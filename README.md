# 3DFoundationModel

## Environment

```
conda update -n base -c conda-forge conda
conda create -n 3DFM
conda activate 3DFM
conda install conda-forge::transformers
```

Check cuda version in your device:
```
nvcc --version
```

Install PyTorch based on your cuda version from [official website](https://pytorch.org/get-started/locally/).

## Dataset

Download NAVI_v1.5 dataset
```
mkdir data
cd data

# Download (v1.5) 
wget https://storage.googleapis.com/gresearch/navi-dataset/navi_v1.5.tar.gz

# Extract
tar -xzf navi_v1.5.tar.gz
```
