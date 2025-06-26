from GeneExpressionData import GeneExpressionData
from PredictionModel import PredictionModel
import time

def main():
    start = time.time()
    model = PredictionModel(folder='Yeast_NN_GFP/Dev',modelName='TestModel',numFolds=4,datasetMode='2007')
    model.trainFold(3,numBatches=5000)
    print(f'Training took {(time.time() - start)/60} minutes')

if __name__ == "__main__":
    main()