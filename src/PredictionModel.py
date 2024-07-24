import pandas as pd
import numpy as np
import os

from GeneOntology.GOParser import GOParser

class PredictionModel():
    '''
    PredictionModel is a class uses gene expression data to predict the functional relationship between gene pairs in the context of gene ontology terms (GO terms). This class contains the workflow to organize model, train neural networks, and make predictions of the functions of genes or pairs of genes.
    '''

    def __init__(self,
                 modelName:str,folder='.',ontologyDate:str = '2022-06-15',numFolds:int=4,
                 inputData:str='x',outputTerms:str='b',addtionalOutputs:list[str]=[],
                 hiddenStructure:str='500x200x100',activationFunc:str='ReLU',lossFunc:str='BCE',
                 lr:float=0.01,batchSize:int=50,momentum:float=0.9,
                 weightDecay:float=0.0,inputDropout:float=None,hiddenDropout:float=None,
                 goldStandardFile:str=None
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

            -hiddenStructure - string of integers separated by 'x' specifying the number of nodes in each hidden layer of the neural network
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
        if not os.path.exists(f'{folder}/{modelName}'):
            os.mkdir(f'{folder}/{modelName}')

        # Initalize model parameters to fields
        self.numFolds = numFolds

        
        self.parser = GOParser(ontologyDate)
        self.terms = self.parser.getSlimLeaves(roots=outputTerms)


        # Intialization of gold standard and gene folds
        self.goldStandard = set()
        for id, genes in self.terms: 
            self.goldStandard = self.goldStandard.union(genes)

        if goldStandardFile is not None:
            gsFile = goldStandardFile
        else:
            gsFile = f'{folder}/{modelName}/GoldStandardGenes.csv'
        if os.path.exists(gsFile):
            self.geneFolds = self.loadGeneFoldsFromFile(gsFile)
        else:
            self.geneFolds = self.createNewGeneFolds(gsFile)

        # Initialize model input data

        

    def loadGeneFoldsFromFile(self,fileName:str) -> list[set[str]]:
        ''' Loads k-folds gene splits from a csv file '''

        geneFolds = [set() for i in range(self.numFolds)]
        table = pd.read_csv(fileName)
        for _, row in table.iterrows():
            geneFolds[row['Fold']].add(row['Gene'])

        return geneFolds
    
    def createNewGeneFolds(self,fileName:str) -> list[set[str]]:
        ''' Creates a new set of k-fold gene folds and saves them to a csv file '''

        # Shuffle list of gold standard genes
        goldStandard = list(self.goldStandard)
        np.random.shuffle(goldStandard)

        table = []
        # Assign each gene to numFolds evenly sized folds, then save it to a csv file
        
        for i in range(len(goldStandard)):
            table.append([goldStandard[i],i % self.numFolds])
        table.sort(key=lambda x: x[1])
        pd.DataFrame(table,columns=['Gene','Fold']).to_csv(fileName,index=False)
            
        # Load the gene folds from the file that was just created
        return self.loadGeneFoldsFromFile(fileName)
    
    
    # ---------------------------------------------------------------------------------- #
    # ----------------------------------- Unit Tests ----------------------------------- #
    # ---------------------------------------------------------------------------------- #
    def hasIndependentGeneFolds(self):
        ''' 
        Tests if a model's gene folds are indepdent of one another, that is, they share no overlap in their set of genes
        '''
        for i in range(self.numFolds):
            for j in range(i+1,self.numFolds):
                if len(self.geneFolds[i] & self.geneFolds[j]) != 0:
                    return False
        return True
    
    def geneFoldsCoverGoldStandard(self):
        ''' 
        Tests if a model's gene folds cover the entire gold standard set of genes
        '''
        return len(set.union(*self.geneFolds)) == len(self.goldStandard)
    
    def geneFoldsHaveEqualSize(self):
        '''
        Tests if a model's gene folds are roughly equal in size. This means that each fold is within 1 gene of all other gene folds.
        '''

        for i in range(self.numFolds):
            for j in range(i+1,self.numFolds):
                if abs(len(self.geneFolds[i]) - len(self.geneFolds[j])) > 1:
                    return False
        return True
            


        

