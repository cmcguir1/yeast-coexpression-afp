from PredictionModel import PredictionModel

def main():
    model = PredictionModel(folder='~',modelName='Demo',numFolds=4,datasetMode='2007')
    model.trainNetworks()
    model.evaluatePairwisePerformance()
    model.constructFunctionalRelationshipGraph()
    model.singleGeneRankings()

if __name__ == "__main__":
    main()