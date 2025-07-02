from GeneExpressionData import GeneExpressionData
from PredictionModel import PredictionModel
import time

def main():
    start = time.time()
    model = PredictionModel(folder='Dev',modelName='TestModel_Untrained',numFolds=4,datasetMode='2007')
    # model.constructFunctionalRelationshipGraph()
    model.singleGeneRankings()
    # for i in range(4):
    #     model.trainFold(i,numBatches=1)
    print(f'Training took {(time.time() - start)/60} minutes')

if __name__ == "__main__":
    main()