from GeneExpressionData import GeneExpressionData
from PredictionModel import PredictionModel

def main():
    model = PredictionModel(folder='Yeast_NN_GFP/Dev',modelName='TestModel',numFolds=4,datasetMode='2007')

if __name__ == "__main__":
    main()