from GeneExpressionData import GeneExpressionData
from PredictionModel import PredictionModel
import time
import numpy as np
from GeneOntology.GOParser import GOParser

def main():
    start = time.time()
    model = PredictionModel(folder='Dev',modelName='TestModel_79',numFolds=4,datasetMode='2007',ontologyDate='2007-04-01')
    model.trainNetworks()
    model.evaluatePairwisePerformance()
    model.constructFunctionalRelationshipGraph()
    model.singleGeneRankings()

    print('Time taken:', (time.time() - start)/60, 'minutes')
    

if __name__ == "__main__":
    main()