# Jupyter notebook for processing SSRL2-1 Pilatus 100K data
If this is not installed in the account you are using on the BL2-1 computer:

Open anaconda prompt in the folder you want to run processing and run the following commands to create a conda environment with the necessary dependencies:
```
> git clone https://github.com/yue-here/BL21-processing
  (or download the repo from that URL)
> conda env create -f bl21jupyter.yml
> conda activate bl21jupyter
> jupyter notebook
<open the .ipynb file in the browser interface>
```

For testing away from the beamline, you can use the tutorial branch which contains test data:
https://github.com/yue-here/BL21-processing/tree/tutorial
