# Does SpatioTemporal information benefit Two video summarization benchmarks?

This is the repository for this project on the analysis as to whether video summarization datasets are affected greatly by temporal perturbations and the performance temporally invariant models achieve. It contains the code for each of the experiments, the configurations behind each experiment, as some code to read the results, analyse them and generate some of the plots given in the paper.

# Weights and replication

To retrieve the exact results made from the paper, the weights are not given with the repository, please contact Aashutosh.ganesh@maastrichtuniversity.nl for the specific weights for replication. 
# Acknowledgements

This work wouldn't have been possible without invaluable contributions made by other researchers within video Summarization such as

    1. Fajtl et al with VASNet
    2. DSNet
    3. Apostolidis et al with PGL-SUM



# Instructions

## Data

Download the pre-processed TVSum and SumMe datasets alongside the original TVSum annotation file from this [link](). Place the two .h5 files in Data/h5datasets and place the .m file Utils/ folders respectively.

## Run an experiment 
An experiment can be run with the following commands

 ```bash
python  train-split.py --config_path path/to/config
```

For the Multi-Layer perceptron experiments described in the paper, you can use the following execution

```bash
python mlp-split.py --config_path path/to/config
```

To evaluate the model, it can be done by running the following scripts 
```bash
python  evaluate-split.py --config_path path/to/config --delete_weights
```

## Important notes

- The experiments are always configured to run on the first gpu, to adjust that please adjust the ```python device``` variable in all of the scripts



