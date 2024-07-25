import numpy as np
import scipy as sp
import pandas as pd
import torch
import os

class GeneExpressionData():
    '''
    This class is responsible for taking high-throughput yeast gene expression data and converting it into usable input features for our the PredictionModel. It's primary function is creating a lookup table that maps pairs of genes to the normalized pearson correlations between their experimental conditions in each of the input datasets.
    '''

    def __init__(self,genes,datasetMode='all',datasetsFile=None):
        '''
        Initialization of GeneExpressionData object.

        Arguments:
            -genes - list of genes that will be included in the Pearson correlation lookup tables
            -datasetMode - string that specifies how the object will decide which datasets are included in the final gene expression compendium. Options are:
                -'all' - include all datasets included in the Yeast_NN_GFP/data/GeneExpression/Datasets/ directory
                -'file' - include only datasets specified in the datasetsFile argument
                -'2007' | 'Comparison' - include only that gene expression datasets used by MEFIT, SPELL, and bioPIXIE in Hess et al. 2009 (https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1000407#s5)
            -datasetsFile - text file with a list of dataset file names that should be included in the gene expression compendium. Each file name should be on a different line, and all file should be located in the Yeast_NN_GFP/data/GeneExpression/Datasets/ directory
            
        '''
        packageDir = os.path.dirname(__file__) + '/..'

        if datasetMode in ['all','All']:
            # Load in all files from /data/GeneExpression/Datasets/
            files = os.listdir(f'{packageDir}/data/GeneExpression/Datasets/')
        elif datasetMode in ['file','File']:
            # Load in all files from datasetsFile (if datasetsFile exists)
            if not os.path.exists(datasetsFile): raise Exception(f'File {datasetsFile} does not exist')
            with open(datasetsFile,'r') as f:
                files = [line.rstrip() for line in f]
        
            