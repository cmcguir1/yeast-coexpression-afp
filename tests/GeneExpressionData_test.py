import pytest
import os 
from src.GeneExpressionData import GeneExpressionData

testFile = os.path.dirname(__file__) +'../data/GeneExpression/TestDatasets.txt'

genes = ['A','B','C']
geneExp = GeneExpressionData(genes,datasetMode='file',datasetsFile=)