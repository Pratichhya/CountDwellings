# Counting Dwellings without detecting them

Counting without detection: applicability of Gaussian Density maps. A summer task in coordination with Prof. Stefan Lang from University of Salzburg was conducted.
The scripts are inspired from: https://github.com/NeuroSYS-pl/objects_counting_dmap.git

The entire script can be run by: python main.py

### Notebook:

1. Documentation_T2.ipnyb: It provides an overview of the project including dataset used and method adopted.
2. npy_generator.ipnyb: It provides scripts to prepare npy file by performing basic preprocessing steps. Path to that npy file can then be updated in data_loader.py which will then show the path to the dataset.

### Remarks:

- Data is not provided in this repository as it a private dataset was used for research purpose only. You can use any satellite image or even computer vision dataset that has 3 bands and corresponding label (point raster).
  Or even higher band dataset can be used with slight updating of the input layer to the model
