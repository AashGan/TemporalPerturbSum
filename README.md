# Does SpatioTemporal information benefit Two video summarization benchmarks?

This is the repository for this project on the analysis as to whether video summarization datasets are affected greatly by temporal perturbations and the performance temporally invariant models achieve. It contains the code for each of the experiments, the configurations behind each experiment, as some code to read the results, analyse them and generate some of the plots given in the paper.

# Weights and replication

To retrieve the exact results made from the paper, please contact Aashutosh.ganesh@maastrichtuniversity.nl for the specific weights for replication. Due to differences in CUDA versions, pytorch versions and the type of GPU, there will be slight differences in the end results from training. 

# Acknowledgements

This work wouldn't have been possible without invaluable contributions made by other researchers within video Summarization such as

1. [VASNet](https://github.com/ok1zjf/VASNet/tree/master)
2. [DSNet](https://github.com/li-plus/DSNet)
3. [PGLSUM](https://github.com/e-apostolidis/PGL-SUM/tree/master/model)



# Instructions
## Setup
Run the following command to setup the environment for the experiments. Highly recommended to set up a venv or conda environment for these experiments.

```bash
pip install -r requirements.txt
```
## Data
The preprocessed dataset as provided by the paper "Video summarization with long short term memory" alongside the annotations from the original publication of the TVSum dataset 
Download the pre-processed TVSum and SumMe datasets alongside the original TVSum annotation file from this [link](https://drive.google.com/drive/folders/1ROGe1ifXWwzMJKY1SYPsHAHFwpNJWabS?usp=sharing). Place the two .h5 files in Data/h5datasets and place the .m file Utils/ folders respectively.

## Run an experiment 
An experiment can be run with the following commands

 ```bash
python  train-split.py --config_path path/to/config
```

For the Multi-Layer perceptron experiments described in the paper, you can run the following

```bash
python mlp-split.py --config_path path/to/config
```

To evaluate the model, it can be done by running the following scripts 
```bash
python  evaluate-split.py --config_path path/to/config --delete_weights
```

## Important notes

- The experiments are always configured to run on the first gpu, to adjust that please adjust the ```python device``` variable in all of the scripts or set ```CUDA_VISIBLE_DEVICES``` to your preferred device prior to running any of the scripts



