import pandas as pd
import numpy as np
import os
import torch
import math
import time

from GeneOntology.GOParser import GOParser
from GeneExpressionData import GeneExpressionData
from FeedForwardNetwork import FeedForwardNetwork

class PredictionModel():
    '''
    PredictionModel is a class uses gene expression data to predict the functional relationship between gene pairs in the context of gene ontology terms (GO terms). This class contains the workflow to organize model, train neural networks, and make predictions of the functions of genes or pairs of genes.
    '''

    def __init__(self,
                 modelName:str,folder='.',ontologyDate:str = '2007-01-01',numFolds:int=4,
                 inputData:str='x',outputTerms:str='b',addtionalOutputs:list[str]=[],
                 hiddenStructure:list[int]=[500,200,100],activationFunc=torch.nn.ReLU(),
                 goldStandardFile:str=None,datasetMode:str='2007',datasetsFile:str=None
                 ):
        '''
        Initialize the PredictionModel object.

        Arguments:
            -modelName - name of prediction model. A folder with this name will be create to store all data related to this model
            -folder - the folder your model will be save. By default, model is saved to your current working directory
            -ontologyDate - date Gene Ontology annotations and DAG will be pulled from. Default is 2007.
            -numFolds - number of folds used for k-fold cross validation.

            -inputData - string of single letter flags specifiying what kinds of data model will use as features. Each flag is specified below:
                x - gene expression data
                l - subcellular localization data from Huh et al. 2003
                g - genomic interaction data from bioGRID
                p - physical interaction data from bioGRID
            -outputTerms - string of single letter flags specifiying which terms from the GO slim the model will be trained to predict. Each flag is specified below:
                b - biological process
                c - cellular component
                m - molecular function
            -addtionalOutputs - list of additional GO terms to include in the model predictions

            -hiddenStructure - string of integers separated by 'x' specifying the number of nodes in each hidden layer of the neural network
            -activationFunc - the activation function to used in each hidden layer of the neural networks
            -lossFunc - loss function used to train neural network. By default, the model uses a binary cross entropy with logits loss function.

            
            -goldStandardFile - file path to csv file containing custom k-fold split of yeast genes. If None, a deafult k-fold split name GoldStandardGenes.csv will be created in the model folder
            -datasetMode - string passed to GeneExpressionData object that specifies how the model will decide which datasets are included in the
            -datasetsFile - if datasetMode is 'file', this arguument species the file path to the text file containing the list of gene epxression datasets to use for input features


        '''

        # Check if model folder already exists
        self.modelFolder = f'{folder}/{modelName}'
        if not os.path.exists(self.modelFolder):
            os.mkdir(self.modelFolder)

        # Create folder to store pairwise performance of each fold
        if not os.path.exists(f'{self.modelFolder}/Pairwise'):
            os.mkdir(f'{self.modelFolder}/Pairwise')

        # Initalize model parameters to fields
        self.numFolds = numFolds

        # Create Gene Ontology parser and get GO slim terms
        self.parser = GOParser(ontologyDate)
        self.terms = self.parser.getSlimLeaves(roots=outputTerms)


        # Intialization of set of all genes, gold standard, and gene folds
        self.allGenes = self.parser.allGenes()

        self.goldStandard = set()
        for _, genes in self.terms: 
            self.goldStandard = self.goldStandard.union(genes)

        if goldStandardFile is not None:
            gsFile = goldStandardFile
        else:
            gsFile = f'{folder}/{modelName}/GoldStandardGenes.csv'
        if os.path.exists(gsFile):
            self.geneFolds = self.loadGeneFoldsFromFile(gsFile)
        else:
            self.geneFolds = self.createNewGeneFolds(gsFile)

        # Initialize model input data
        self.expressionData = GeneExpressionData(genes=self.allGenes,datasetMode=datasetMode,datasetsFile=datasetsFile)

        # Initalize model neural networks
        self.networks = []
        for i in range(numFolds):
            network = FeedForwardNetwork(inputSize=self.expressionData.numDatasets,
                                                    outputSize=len(self.terms),
                                                    hiddenLayers=hiddenStructure,activation=activationFunc)
            # If network has been previously trained, load in weights
            if(os.path.exists(f'{self.modelFolder}/Network_fold{i}.pth')):
                network.load_state_dict(torch.load(f'{self.modelFolder}/Network_fold{i}.pth'))
            self.networks.append(network)
    
    def trainFold(self,fold:int,numBatches:int=600000,batchSize=50,lr=0.01,momentum=0.9):
        '''
        Trains a single fold of the model's neural network
        
        Arguments:
            -fold - the fold number to train
            -numBatches - the number of mini-batches to use for training
            -batchSize - the number of gene pairs in each mini-batch
            -lr - learning rate for stochastic gradient descent optimizer
            -momentum - momentum for stochastic gradient descent optimizer
        '''
        # Initialize neural network, loss function, and optimizier
        network = self.networks[fold]
        network.train()
        
        lossFunction = torch.nn.BCEWithLogitsLoss(reduction='mean')
        device = 'cuda:0' if torch.cuda.is_available() and batchSize > 200 else 'cpu'
        optimizer = torch.optim.SGD(network.parameters(),lr=lr,momentum=momentum)
        network.to(device)

        # Get training and testing gene split for fold, then turn each into sets of positive and negative gene pairs
        trainGenes, testGenes = self.getDataSplit(fold)
        trainPosPairs, trainNegPairs = self.splitPosNegPairs(trainGenes)
        testPosPairs, testNegPairs = self.splitPosNegPairs(testGenes)

        np.random.shuffle(trainPosPairs); np.random.shuffle(trainNegPairs)
        np.random.shuffle(testPosPairs); np.random.shuffle(testNegPairs)

        # Initialize loss table to keep track of training and tessting loss over time
        lossTable = []
        runningLoss = 0.0

        for i in range(numBatches):
            batch =self.createBatch(i,trainPosPairs,trainNegPairs,batchSize)
            features = self.featuresVector(batch)
            labels = self.labelsVector(batch)
            features = features.to(device)
            labels = labels.to(device)
            
            # Reset gradients, perform forward pass, calculate loss, and backpropagate

            optimizer.zero_grad()
            outputs = network(features)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()
            

            runningLoss += loss.item()

            # Every 1000 batches, evaluate the model on the test set and save the loss
            if i % 1000 == 0:
                with torch.no_grad():
                    testBatch =self.createBatch(i//1000,testPosPairs,testNegPairs,batchSize*100)
                    testFeatures = self.featuresVector(testBatch)
                    testLabels = self.labelsVector(testBatch)

                    testOutputs = network(testFeatures)
                    testLoss = lossFunction(testOutputs,testLabels)

                    if(i == 0):
                        runningLoss *= 1000
                    lossTable.append((i,runningLoss,testLoss.item() * 1000))
                    runningLoss = 0.0
        
        # Save network's weights and the loss table in model's folder
        torch.save(network.state_dict(),f'{self.modelFolder}/Network_fold{fold}.pth')
        pd.DataFrame(lossTable,columns=['Batch','Train Loss','Test Loss']).to_csv(f'{self.modelFolder}/Loss_fold{fold}.csv',index=False)

    def evaluateFoldPerformance(self,fold:int):
        '''
        Evaluates the pairwise performance of a fold's neural network on the test set.
        '''

        with torch.no_grad():

            network = self.networks[fold]
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            network.to(device)

            # Get testing gene split for fold, then it sets of positive and negative gene pairs (only the negative pair will  be used)
            _, testGenes = self.getDataSplit(fold)
            _, testNegPairs = self.splitPosNegPairs(testGenes)

            performance = []

            

            for i, (term, posGenes) in enumerate(self.terms):
                print(term)
                posPairs = np.array([[geneA, geneB] for geneA in posGenes for geneB in posGenes if geneA != geneB],dtype='U10')
                np.random.shuffle(posPairs)
                posPairs = posPairs[:min(10000,len(posPairs))]
                np.random.shuffle(testNegPairs)
                negPairs = testNegPairs[:min(10*len(posPairs),len(testNegPairs))]

                termPairs = np.concatenate((posPairs,negPairs),axis=0)

                features = self.featuresVector(termPairs)
                features = features.to(device)
                outputs = network(features)

                confidenceValues = outputs[:,i].cpu().numpy().flatten()
                labels = np.concatenate((np.ones(len(posPairs)),-1*np.ones(len(negPairs))),axis=0)

                temp = np.stack([confidenceValues,labels],axis=1)
                termRanking = np.concatenate([termPairs,temp],axis=1,dtype=object)

                termRanking = PredictionModel.thresholdSweep(termRanking)
                termRanking.to_csv(f'{self.modelFolder}/Pairwise/{term.replace(":","-")}_Fold{fold}_.csv',index=False)

                avgPrecision, auc = PredictionModel.AveragePrecisionAndAUC(termRanking)

                performance.append((term,avgPrecision,auc))
            
            pd.DataFrame(performance,columns=['Term','Average Precision','AUC']).to_csv(f'{self.modelFolder}/Pairwise/PerformanceSummary_Fold{fold}.csv',index=False)


    #----------------------------------------------------------------------------------#
    #------------------------------- Helper Functions ---------------------------------#   
    #----------------------------------------------------------------------------------#

    def thresholdSweep(table:np.ndarray) -> pd.DataFrame:
        '''
        Given a table of gene pairs, confidence values, and labels, this method performs a threshold sweep to generate the total operating characteristic of the predictions
        
        Arguments:
            -table - a numpy array of shape (n,4) where n is the number of gene pairs. Each row contains a gene pair, its confidence value, and its label. The columns are as follows:
                0 - Gene A
                1 - Gene B
                2 - Confidence Value
                3 - Label (1 for positive, 0 for negative)
        '''


        # Sort the table by confidence values in descending order
        table = table[table[:,2].argsort()[::-1]]
        print(table)

        truePositives = 0
        falsePositives = 0
        trueNegatives = sum(table[:,3] == -1)
        falseNegatives = sum(table[:,3] == 1)

        def calculateStatistics():
            falsePositiveRate = falsePositives / (falsePositives + trueNegatives)
            recall = truePositives / (truePositives + falseNegatives)
            precision = truePositives / (truePositives + falsePositives)
            return np.array([falsePositiveRate,recall,precision])


        confusionMatrixStatistics = np.ndarray((len(table),3),dtype=object)

        for i,row in enumerate(table):
            if row[3] == 1:
                truePositives += 1
                falseNegatives -= 1
            elif row[3] == -1:
                falsePositives += 1
                trueNegatives -= 1

            confusionMatrixStatistics[i] = calculateStatistics()

        ranking = np.concatenate((table,confusionMatrixStatistics),axis=1)
        
        return pd.DataFrame(ranking,columns=['Gene A','Gene B','Confidence Value','Label',
                                            'False Positive Rate','Recall','Precision'])
    
    def AveragePrecisionAndAUC(table:pd.DataFrame) -> float:
        '''
        Calculates the auc for ROC curve given a table of gene pairs, confidence values, and labels

        Arguments:
            -table - a pandas DataFrame with columns 'Gene A', 'Gene B', 'Confidence Value', and 'Label'
        '''

        # Sort the table by confidence values in descending order
        table = table.sort_values(by='Confidence Value', ascending=False)

        table = table.loc[table['Label'] != 0]

        
        recall = table['Recall'].values
        auc = np.mean(recall)

        precision = table['Precision'].values
        for i in range(len(precision)-1):
            if precision[i] < precision[i+1]:
                precision[i+1] = precision[i]
        
        averagePrecision = np.mean(precision)

        return averagePrecision, auc





    def loadGeneFoldsFromFile(self,fileName:str) -> list[set[str]]:
        ''' Loads k-folds gene splits from a csv file '''

        geneFolds = [set() for i in range(self.numFolds)]
        table = pd.read_csv(fileName)
        for _, row in table.iterrows():
            geneFolds[row['Fold']].add(row['Gene'])

        return geneFolds
    
    def createNewGeneFolds(self,fileName:str) -> list[set[str]]:
        ''' Creates a new set of k-fold gene folds and saves them to a csv file '''

        # Shuffle list of gold standard genes
        goldStandard = list(self.goldStandard)
        np.random.shuffle(goldStandard)

        table = []
        # Assign each gene to numFolds evenly sized folds, then save it to a csv file
        
        for i in range(len(goldStandard)):
            table.append([goldStandard[i],i % self.numFolds])
        table.sort(key=lambda x: x[1])
        pd.DataFrame(table,columns=['Gene','Fold']).to_csv(fileName,index=False)
            
        # Load the gene folds from the file that was just created
        return self.loadGeneFoldsFromFile(fileName)
    
    def getDataSplit(self,fold:int) -> tuple[np.ndarray,np.ndarray]:
        '''
        Returns the tuple of the training and testing gene sets (in that order) for a given fold
        '''
        testGenes = []
        trainGenes = []
        for i in range(self.numFolds):
            if i == fold:
                testGenes = list(self.geneFolds[i])
            else:
                trainGenes += list(self.geneFolds[i])

        return np.array(trainGenes,dtype='U10'),np.array(testGenes,dtype='U10')

    def splitPosNegPairs(self,genes:list[str]):
        '''
        Given a set of genes, this method returns a tuple of two sets. The first set contains all positive gene pairs (pairs of genes that are co-annotated to at least one GO term) and the second set contains all negative gene pairs (pairs of genes that are not co-annotated to any GO terms and whose smallest co-annotated term has less than 10% of the yeast genome annotated to it).
        '''
        posPairs = []
        negPairs = []

        geneSet = set(genes)

        smallestCommonAncestorFile = os.path.dirname(__file__) + '/../data/Pairs_SmallestCommonAncestor.npy'
        allPairs = np.load(smallestCommonAncestorFile,allow_pickle=True)
        smallestCommonAncestorThreshold = 0.1 * len(self.allGenes) # the maximum number of annotations a pair's smallest common ancestor term can have to be considereed a negative pair

        # Iterate through all pairs of genes
        for pair in allPairs:
            
            geneA = pair[0]
            geneB = pair[1]
            smallestCommonAncestor = pair[2]
            if geneA in geneSet and geneB in geneSet:
                if self.coannotated(geneA,geneB):
                    posPairs.append((geneA,geneB))
                elif smallestCommonAncestor < smallestCommonAncestorThreshold:
                    negPairs.append((geneA,geneB))

        return np.array(posPairs,dtype='U10'), np.array(negPairs,dtype='U10')

    def coannotated(self,geneA:str,geneB:str) -> bool:
        '''
        Predicate that returns true if geneA and geneB are co-annotated to at least one GO term in the model's slim
        '''
        for _, termGenes in self.terms:
            if geneA in termGenes and geneB in termGenes:
                return True
        return False
    
    def createBatch(self,i,posPairs,negPairs,batchSize:int):
        '''
        Creates a mini-batch of gene pairs containing an equal number of positive and negative pairs
        '''
        # posBatch = posPairs[np.random.choice(len(posPairs),size=math.ceil(batchSize/2),replace=False)]
        # negBatch = negPairs[np.random.choice(len(negPairs),size=math.floor(batchSize/2),replace=False)]
        posBatch = posPairs[i%len(posPairs):(i+batchSize//2)%len(posPairs)]
        negBatch = negPairs[i%len(negPairs):(i+batchSize//2)%len(negPairs)]

        return np.concatenate((posBatch,negBatch),axis=0)
    
    def featuresVector(self,batch:np.ndarray) -> torch.Tensor:
        '''
        Given a batch of gene pairs, this method return a 2D vector where each row is the feature vector for a gene pair
        '''

        indices = torch.IntTensor([self.expressionData.genePairIndex(pair[0],pair[1]) for pair in batch])

        features = self.expressionData.correlationDictionary.index_select(0,indices).float()

        # features = np.zeros(shape=(len(batch),self.expressionData.numDatasets),dtype=np.float32)
        # for i, pair in enumerate(batch):
        #     geneA, geneB = pair
        #     features[i] = self.expressionData.similarityVector(geneA,geneB)

        return features
    
    def labelsVector(self,batch:np.ndarray) -> torch.Tensor:
        '''
        Given a batch of gene pairs, this method returns a 2D vector where each row is the label vector for a gene pair
        '''
        labels = []
        for pair in batch:
            geneA, geneB = pair
            label = []
            for _, termGenes in self.terms:
                if geneA in termGenes and geneB in termGenes:
                    label.append(1)
                else:
                    label.append(0)
            labels.append(label)
        return torch.tensor(labels,dtype=torch.float32)

    # ---------------------------------------------------------------------------------- #
    # ----------------------------------- Unit Tests ----------------------------------- #
    # ---------------------------------------------------------------------------------- #
    def hasIndependentGeneFolds(self):
        ''' 
        Tests if a model's gene folds are indepdent of one another, that is, they share no overlap in their set of genes
        '''
        for i in range(self.numFolds):
            for j in range(i+1,self.numFolds):
                if len(self.geneFolds[i] & self.geneFolds[j]) != 0:
                    return False
        return True
    
    def geneFoldsCoverGoldStandard(self):
        '''  Tests if a model's gene folds cover the entire gold standard set of genes '''
        return len(set.union(*self.geneFolds)) == len(self.goldStandard)
    
    def geneFoldsHaveEqualSize(self):
        '''
        Tests if a model's gene folds are roughly equal in size. This means that each fold is within 1 gene of all other gene folds.
        '''

        for i in range(self.numFolds):
            for j in range(i+1,self.numFolds):
                if abs(len(self.geneFolds[i]) - len(self.geneFolds[j])) > 1:
                    return False
        return True
            


        

