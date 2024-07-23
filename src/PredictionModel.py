import pandas as pd
import numpy as np
import os

class PredictionModel():
    '''
    PredictionModel is a class uses gene expression data to predict the functional relationship between gene pairs in the context of gene ontology terms (GO terms). This class contains the workflow to organize model, train neural networks, and make predictions of the functions of genes or pairs of genes.
    '''

    def __init__(self,
                 modelName:str,folder='.',ontoDate:str = '2022-06-15',numFolds:int=4,
                 inputData:str='x',outputTerms:str='b',addtionalOutputs:list[str]=[],
                 networkStructure:str='500x200x100',activationFunc:str='ReLU',lossFunc:str='BCE',
                 lr:float=0.01,batchSize:int=50,momentum:float=0.9,
                 weightDecay:float=0.0,inputDropout:float=None,hiddenDropout:float=None
                 ):
        '''
        Initialize the PredictionModel object.

        Arguments:
            -modelName - name of prediction model. A folder with this name will be create to store  all data related to this model
            -folder - the folder your model will be save. By default, model is saved to your     current working directory
            -ontoDate - date Gene Ontology annotations and DAG will be pulled from. Default is 2022.
            -numFolds - number of folds used for k-fold cross validation.

            -inputData - string of single letter flags specifiying what kinds of data model will use as features. Each flag is specified below:
                x - gene expression data
                l - subcellular localization data from Huh et al. 2003
                g - genomic interaction data from bioGRID
                p - physical interaction data from bioGRID
            -outputTerms - string of single letter flags specifiying which terms from the GO slim the model will be trained to predict. Each flag is specified below:
                b - biological process
                c - cellular component
                m - molecular function
            -addtionalOutputs - list of additional GO terms to include in the model predictions

            -networkStructure - string of integers separated by 'x' specifying the number of nodes in each hidden layer of the neural network
            -activationFunc - the activation function to used in each hidden layer of the neural networks
            -lossFunc - loss function used to train neural network. By default, the model uses a binary cross entropy with logits loss function.

            lr - learning rate hyperparameter
            batchSize - mini-batch size hyperparameter
            momentum - momentum hyperparameter
            weightDecay - scalar hyperparameter for L2 weight regularization

            inputDropout - dropout rate for input layer
            hiddenDropout - dropout rate for hidden layers

        '''

        # Check if model folder already exists
        if os.path.exists(f'{folder}/{modelName}'):
            os.mkdir(f'{folder}/{modelName}')

        

