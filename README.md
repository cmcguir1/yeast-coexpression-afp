# yeast-coexpression-afp
This repository contains the scripts needed to run the co-expression-bassed automated function prediction model in yeast described in McGuire and Hibbs 2025. The model trains four feedforward neural networks to predict the co-annotations of pairs of genes to 79 Gene Ontology biological process terms from the co-expression profiles of the pair of genes in 113 gene expression datasets. Once these networks are trained, our model can aggregate its pairwise predictions to rank individual genes by their predicted involvement in each of the 79 GO terms.

## About
### Metholodogy Overview
The bulk of our methodology is implemented in our `PredictionModel` class in `src/PredictionModel.py`. Objects of this class contain a `GeneExpressionData` object to calculate the co-expression profiles of pairs of genes used as input features, a `GOParser` object to determine the Gene Ontology annotations of yeast genes used as labels, and `FeedForwardNetwork` objects that are PyTorch neural networks.

To make single-gene functional predictions with our method, follow these steps:
- Instantiate a `PredictionModel` object
- Train the model's cross-validated feedforward networks with the `trainNetworks` object
- Evaluate each fold of the model's performance on training and testing data with `evaulatePairwisePerformance` method
- Make predictions for all pairs of genes in the yeast genome with the `constructFunctionalRelationshipGraph` method
- Aggreate pairwise predictions from functional relationship graph to rank individual genes by their predicted involvement in 79 GO terms with `singleGeneRankings` method
### Demo
We have provided a demo in `src/demo.py` which performs all of the steps in the list above with the deafult hyperparameters and training time described in McGuire and Hibbs 2025.

## Usage
### Setting up virtual environment
This project uses virtual environments with conda to manage package dependencies. To initalize the virtual environment needed to run this project, use the following commands from within the yeast-coexpression-afp directory:
```
conda env create -f environment.yml
conda activate yeast-coexpression-afp
```

### Downloading datasets
To run this project, you need to download the gene expression and Gene Ontology used to create our model's input features and training labels. This data is available for download on Zenodo. To prepare this repository's data, complete the following step:

1. Download `data.tar.gz` from this [Zenodo link](https://doi.org/10.5281/zenodo.15831976)
2. Place this file in the top level of the `yeast-coexpression-afp` repository directory
3. Run `tar -xzvf data.tar.gz` from the `yeast-coexpression-afp` directory. This will extract the tar ball file into a `data` directory with all files already organized into their respective subfolder.

## Original Implementation
The original implementation of the methodolgy found in McGuire and Hibbs 2025 can be found in [yeast-coexpression-afp-original](https://github.com:cmcguir1/yeast-coexpression-afp-original). The original implementation of these algorithms were very computationally inefficient. This new repository was created to provide computationally efficient implementations of our methodologies and an easy to set up virtual environment to improve the reproducability of our results.
