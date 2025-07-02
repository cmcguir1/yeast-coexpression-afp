from PredictionModel import PredictionModel

def main():
    model = PredictionModel(folder='~',modelName='Demo',numFolds=4,datasetMode='2007')
    for i in range(4):
        model.trainFold(i,numBatches=600_000)
        model.evaluateFoldPerformance(i)
    model.constructFunctionalRelationshipGraph
    model.singleGeneRankings()

if __name__ == "__main__":
    main()