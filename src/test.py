from GeneExpressionData import GeneExpressionData
from PredictionModel import PredictionModel
import time

def main():
    start = time.time()
    model = PredictionModel(folder='Dev',modelName='TestModel',numFolds=4,datasetMode='2007')
    # for i in range(2,4):
    #     model.trainFold(i,numBatches=600_000)
    # for i in range(4):
    #     model.evaluateFoldPerformance(i)
    # model.constructFunctionalRelationshipGraph()
    model.singleGeneRankings()
    print(f'Training took {(time.time() - start)/60} minutes')

if __name__ == "__main__":
    main()