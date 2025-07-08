import numpy as np
import pandas as pd
import os
import torch

class GeneExpressionData():
    '''
    This class is responsible for taking high-throughput yeast gene expression data and converting it into usable input features for our the PredictionModel. It's primary function is creating a lookup table that maps pairs of genes to the normalized pearson correlations between their experimental conditions in each of the input datasets.
    '''

    def __init__(self,genes:set[str],datasetMode:str='all',datasetsFile:str=None):
        '''
        Initialization of GeneExpressionData object.

        Arguments:
            -genes - list of genes that will be included in the Pearson correlation lookup tables
            -datasetMode - string that specifies how the object will decide which datasets are included in the final gene expression compendium. Options are:
                -'all' - include all datasets included in the Yeast_NN_GFP/data/GeneExpression/Datasets/ directory
                -'file' - include only datasets specified in the datasetsFile argument
                -'2007' or 'Comparison' - include only that gene expression datasets used by MEFIT, SPELL, and bioPIXIE in Hess et al. 2009 (https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000407#s5)
            -datasetsFile - text file with a list of dataset file names that should be included in the gene expression compendium. Each file name should be on a different line, and all file should be located in the Yeast_NN_GFP/data/GeneExpression/Datasets/ directory
            
        '''
        self.packageDir = os.path.dirname(__file__) + '/..'

    

        geneIndices = pd.read_csv(f'{self.packageDir}/data/AllYeastGenes.csv').to_numpy()
        self.genes = [gene[0] for gene in geneIndices]
        self.geneIndex = {gene[0]: gene[1] for gene in geneIndices}
        self.geneIndexRev = {v: k for k, v in self.geneIndex.items()}

        

        # calculate the total number of pairs in the yeast genome give the number of genes
        n = len(self.genes)
        self.pairsLen = ((n*(n-1)) // 2) + 2 *n


        # Locate datasets files
        if datasetMode in ['all','All']:
            # Load in all files from /data/GeneExpression/Datasets/
            files = os.listdir(f'{self.packageDir}/data/GeneExpression/Datasets/')

        elif datasetMode in ['file','File']:
            # Load in all files from datasetsFile (if datasetsFile exists)
            if not os.path.exists(datasetsFile): raise Exception(f'File {datasetsFile} does not exist')
            with open(datasetsFile,'r') as f:
                files = [line.rstrip() for line in f]

        elif datasetMode in ['2007','Comparison','comparison']:
            # Load in all files from ComparisonDatasets.txt
            with open(f'{self.packageDir}/data/GeneExpression/ComparisonDatasets.txt','r') as f:
                files = [line.rstrip() for line in f]

        # Check if all datasets have a CorrelationDictionary created. If not, create one
        for file in files:
            if not os.path.exists(f'{self.packageDir}/data/GeneExpression/CorrelationDictionaries/{file}_corrDict.npy'):
                self.createCorrelationDictionary(file)

        self.numDatasets = len(files)
        
        # Assemble each dataset's correlation dictionary into a 2D array
        self.correlationDictionary = np.zeros((self.pairsLen,len(files)),dtype=np.float16)
        for i,file in enumerate(files):
            self.correlationDictionary[:,i] = np.load(f'{self.packageDir}/data/GeneExpression/CorrelationDictionaries/{file}_corrDict.npy')

        # Convert the correlation dictionary into a PyTorch tensor for faster access to each gene pair's features
        self.correlationDictionary = torch.from_numpy(self.correlationDictionary.astype(np.float16))


    def createCorrelationDictionary(self,file:str) -> None:
        '''
        Creates a Pearson Correlation dictionary for gene pairs for a given dataset file.

        Arguments:
            -file - name of the dataset file to create the correlation dictionary for
        '''
        print(f'Creating correlation dictionary for {file}, this may take a while...')

        # Read in expression data from file, then filter unnecessary columns and rows
        data = pd.read_csv(f'{self.packageDir}/data/GeneExpression/Datasets/{file}',sep='\t')
        data.drop(columns=['NAME','GWEIGHT'],axis=1,inplace=True) # Drop NAME and GWEIGHT columns
        data.drop(labels=[0],axis=0,inplace=True) # Drop first row (EWEIGHT)

        # Convert data into a dictionary where the keys are gene names and the values are a numpy array of their experimental conditions
        expData = data.to_numpy()
        expDictionary = {}
        for row in expData:
            expDictionary[row[0]] = row[1:].astype(np.float16)

        genesInDataset = list(set(expDictionary.keys()).intersection(self.genes))  # Get genes that are in both the dataset and the genes list

        pairsInDataset = [(geneA,geneB) for geneA in genesInDataset for geneB in genesInDataset if geneA < geneB]
        indices = np.array([self.genePairIndex(geneA,geneB) for geneA,geneB in pairsInDataset],dtype=np.int32)

        # Initialize 1D lookup array for gene pair correlation values

        pairDictionary = np.zeros((self.pairsLen,),dtype=np.float16)

        pearsonCorrelations = np.array([self.customCorrelation(expDictionary[geneA],expDictionary[geneB]) for geneA,geneB in pairsInDataset],dtype=np.float16)

        pairDictionary[indices] = pearsonCorrelations

        print('Normalizing correlation dictionary...')
        np.random.shuffle(pairDictionary)  # Shuffle the pairDictionary to ensure randomness in normalization
        mean = np.mean(pairDictionary[:200000])
        std = np.std(pairDictionary[:200000])
        pairDictionary = (pairDictionary - mean) / std

        # Save the pairDictionary to a file
        np.save(f'{self.packageDir}/data/GeneExpression/CorrelationDictionaries/{file}_corrDict.npy',pairDictionary)

    def customCorrelation(self,vec1:np.ndarray,vec2:np.ndarray) -> float:
        '''

        Calculates the Fisher Z transformed Pearson correlation between two vectors of gene expression data. If this method determines the quality of a pair of vectors to be too low, then the method returns 0 for the Fisher-Z transformed correlation. A pair of vectors is considered to be of low quality if:
            -The number of indicies where either vector has NaN as its value is greater than half the length of the vectors
            -The number of valid indicies is less than 2
            -The standard deviation of the vectors (filtered to only contain values from valid indicies) is 0

        '''

        # Find valid indices (indices where neither vector's value is NaN)
        validIndices = []
        for i in range(len(vec1)):
            if not np.isnan(vec1[i]) and not np.isnan(vec2[i]):
                validIndices.append(i)

        # Return 0 if the number of valid indices is less than 2 or less than half the length of the vectors
        if len(validIndices) < 2 or len(validIndices) < len(vec1) // 2:
            return 0
        
        # Filter vectors to only contain values from valid indices, then check standard devations
        vec1 = vec1[validIndices]
        vec2 = vec2[validIndices]
        if np.std(vec1) == 0 or np.std(vec2) == 0:
            return 0
        
        # Calculate Pearson correlation between vectors, then convert to Fisher Z transformed correlation
        p = np.corrcoef(vec1,vec2)[0,1]
        if p == 1: p = 0.9999 # arctanh is undefined at 1
        elif p == -1: p = -0.9999 # arctanh is undefined at -1
        z = np.arctanh(p)

        return z
            
    def genePairIndex(self,geneA:str,geneB:str) -> int:
        '''
        Returns the index of the pair of genes in the two-key dictionary/ 1D lookup array.
        
        The index is calculated as follows:
            -Each gene recieves a number (a and b) from the geneIndex dictionary, which is the position of the gene in the (alphabetical) sorted list of genes
            -The smaller index becomes the row number (r) and the larger index becomes the column number (c)        
        '''

        a = self.geneIndex[geneA]
        b = self.geneIndex[geneB]
        r = max(a,b)
        c = min(a,b)

        index = (r + (c * len(self.genes))) - ((c*(c-1)) // 2)
        
        return index
    
    def similarityVector(self,geneA:str,geneB:str) -> torch.Tensor:
        '''
        Returns the z-score vector between two genes in all datasets.
        '''
        return self.correlationDictionary[self.genePairIndex(geneA,geneB)]
    
    
    def countExperimentalConditions(self,file:str) -> int:
        '''
        Returns the number of experimental conditions in a dataset file.
        '''
        data = pd.read_csv(f'{self.packageDir}/data/GeneExpression/Datasets/{file}',sep='\t')
        data.drop(columns=['NAME','GWEIGHT'],axis=1,inplace=True)
        return len(data.columns) - 1

        
        
        

        
            