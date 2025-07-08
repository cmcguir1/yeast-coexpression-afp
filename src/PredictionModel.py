import pandas as pd
import numpy as np
import os
import torch


from GeneOntology.GOParser import GOParser
from GeneExpressionData import GeneExpressionData
from FeedForwardNetwork import FeedForwardNetwork

class PredictionModel():
    '''
    PredictionModel is a class uses gene expression data to predict the functional relationship between gene pairs in the context of Gene Ontology terms (GO terms). This class contains the workflow to organize model, train neural networks, and make predictions of the functions of genes or pairs of genes.
    '''

    def __init__(self,
                 modelName:str,folder='.',ontologyDate:str='2007-04-01',numFolds:int=4,
                 outputTerms:str='b',
                 hiddenStructure:list[int]=[500,200,100],activationFunc=torch.nn.ReLU(),
                 geneFoldsFile:str=None,datasetMode:str='2007',datasetsFile:str=None
                 ):
        '''
        Initialize the PredictionModel object.

        Arguments:
            -modelName - name of prediction model. A folder with this name will be create to store all data related to this model
            -folder - the folder your model's folder will be saved in. By default, model is saved to your current working directory
            -ontologyDate - date Gene Ontology annotations and DAG will be pulled from. Default is 2007-04-01
            -numFolds - number of folds used for k-fold cross validation.

            -outputTerms - string of single letter flags specifiying which terms from the GO slim the model will be trained to predict. Each flag is specified below:
                b - biological process
                c - cellular component
                m - molecular function


            -hiddenStructure - list of integers specifying the number of nodes in each hidden layer of the neural network
            -activationFunc - the activation function to used in each hidden layer of the neural networks

            
            -geneFoldsFile - file path to csv file containing custom k-fold split of yeast genes. If None, a deafult k-fold split name GeneFolds.csv will be created in the model folder
            -datasetMode - string passed to GeneExpressionData object that specifies how the model will decide which datasets are included in the
            -datasetsFile - if datasetMode is 'file', this arguument species the file path to the text file containing the list of gene expression datasets to use for input features
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
        self.ontologyDate = ontologyDate


        # Intialization of set of all yeast genes from AllYeastGenes.csv file
        packageDir = os.path.dirname(__file__) + '/..'
        geneIndices = pd.read_csv(f'{packageDir}/data/AllYeastGenes.csv').to_numpy()
        self.allGenes = set([gene[0] for gene in geneIndices])

        # Add all genes annotated to at least on GO term in the GO slim to gold standard set
        self.goldStandard = set()
        for _, genes in self.terms: 
            self.goldStandard = self.goldStandard.union(genes)

        # Check if genesFoldFile is specified, if not, use deafult name. Load this file if it exists, otherwise create a new set of k-folds
        if geneFoldsFile is not None:
            gfFile = geneFoldsFile
        else:
            gfFile = f'{folder}/{modelName}/GeneFolds.csv'

        if os.path.exists(gfFile):
            self.geneFolds = self.loadGeneFoldsFromFile(gfFile)
        else:
            self.geneFolds = self.createNewGeneFolds(gfFile)

        # define the set of agnostic genes - genes that are not annotated to any GO term in the GO slim
        self.agnosticGenes = self.allGenes - self.goldStandard


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

    def trainNetworks(self,numBatches:int=600000,batchSize:int=50,lr:float=0.01,momentum:float=0.9):
        '''
        Trains all folds of the model's neural networks using the trainFold method

        Arguments:
            -numBatches - the number of mini-batches to use for training
            -batchSize - the number of gene pairs in each mini-batch
            -lr - learning rate for stochastic gradient descent optimizer
            -momentum - momentum for stochastic gradient descent optimizer
        '''
        for fold in range(self.numFolds):
            self.trainFold(fold,numBatches,batchSize,lr,momentum)
    
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
        device = 'cuda:0' if torch.cuda.is_available() and batchSize > 200 else 'cpu'
        network.to(device)
        
        lossFunction = torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.SGD(network.parameters(),lr=lr,momentum=momentum)
        

        # Get training and testing gene split for fold, then turn each into sets of positive and negative gene pairs
        trainGenes, testGenes = self.getDataSplit(fold)
        trainPosPairs, trainNegPairs = self.splitPosNegPairs(trainGenes)
        testPosPairs, testNegPairs = self.splitPosNegPairs(testGenes)

        np.random.shuffle(trainPosPairs); np.random.shuffle(trainNegPairs)
        np.random.shuffle(testPosPairs); np.random.shuffle(testNegPairs)

        # Initialize loss table to keep track of training and tessting loss over time
        lossTable = []
        runningLoss = 0.0

        # Training Loop
        for i in range(numBatches):
            optimizer.zero_grad()

            batch = self.createBatch(i,trainPosPairs,trainNegPairs,batchSize)
            features = self.featuresVector(batch).to(device)
            labels = self.labelsVector(batch).to(device)

            # Perform forward pass, calculate loss, and backpropagate
            outputs = network(features)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()
            
            runningLoss += loss.item()

            # Every 1000 batches, evaluate the model on the test set and save the loss
            if i % 1000 == 0:
                with torch.no_grad():
                    testBatch = self.createBatch(i//1000,testPosPairs,testNegPairs,batchSize*100)
                    testFeatures = self.featuresVector(testBatch).to(device)
                    testLabels = self.labelsVector(testBatch).to(device)

                    testOutputs = network(testFeatures)
                    testLoss = lossFunction(testOutputs,testLabels)

                    if(i == 0): runningLoss *= 1000 # Scales the loss of the first batch to that of 1000 batches
                    lossTable.append((i,runningLoss,testLoss.item() * 1000))
                    runningLoss = 0.0
        
        # Save network's weights and the loss table in model's folder
        torch.save(network.state_dict(),f'{self.modelFolder}/Network_fold{fold}.pth')
        pd.DataFrame(lossTable,columns=['Batch','Train Loss','Test Loss']).to_csv(f'{self.modelFolder}/Loss_fold{fold}.csv',index=False)

    def evaluatePairwisePerformance(self):
        '''
        This method run evaluateFoldPerformance on each fold of the model
        '''

        for fold in range(self.numFolds):
            self.evaluateFoldPerformance(fold)

    def evaluateFoldPerformance(self,fold:int):
        '''
        Evaluates the pairwise performance of a fold's neural network on the testing and training gene sets for each GO term. The pairwise results will be saved to csv files
        Arguments:
            -fold - the fold number to evaluate
        '''

        # Helper functions
        
        def filterNegativePairs(pairs:np.ndarray,posGenes:set[str]) -> np.ndarray:
            '''
            Filters a set of gene pairs to only include pairs where both genes are not in the positive gene set
            '''
            return np.array([pair for pair in pairs if pair[0] not in posGenes and pair[1] not in posGenes],dtype='U10')

        def evaluateTermPerformance(termIndex,posGenes,geneSet,negPairs) -> pd.DataFrame:
            '''
            Helper function that evaluates a GO term's performance on a set of genes (the testing genes or training genes)
            '''

            # if a term has less than 5 positive genes in the gene set, the performance evaluation will be uninformative, so return None
            posFoldGenes = posGenes.intersection(geneSet)
            if len(posFoldGenes) < 5:
                return None
            
            # Intialize list of all pairs of genes (in the gene set) that are co-annotated to the term, and take a random sample of at most 10,000 pairs
            posPairs = np.array([[geneA, geneB] for geneA in posFoldGenes for geneB in posFoldGenes if geneA < geneB],dtype='U10')
            np.random.shuffle(posPairs)
            posPairs = posPairs[:min(10000,len(posPairs))]

            # Take a random sample of negative pairs that is at most 10 times the size of the positive pairs
            np.random.shuffle(negPairs)
            negPairs = filterNegativePairs(negPairs,posGenes)
            maxNegPairs = min(10*len(posPairs),len(negPairs))
            negPairs = negPairs[:maxNegPairs]

            # Compute confidence values for each pair of genes
            termPairs = np.concatenate((posPairs,negPairs),axis=0)
            features = self.featuresVector(termPairs).to(device)
            outputs = network(features)
            confidenceValues = outputs[:,termIndex].cpu().numpy().flatten()

            # Create labels for the pairs of genes, where 1 is a positive pair and -1 is a negative pair
            posLabels = np.ones(len(posPairs))
            negLabels = -1*np.ones(len(negPairs))
            labels = np.concatenate((posLabels,negLabels),axis=0)

            # Stack gene pairs, confidence values, and labels, then perform a threshold sweep
            temp = np.stack([confidenceValues,labels],axis=1)
            labeledRanking =  np.concatenate([termPairs,temp],axis=1,dtype=object)
            labeledRanking = pd.DataFrame(labeledRanking,columns=['Gene A','Gene B','Confidence Value','Label'])
            termRanking = PredictionModel.thresholdSweep(labeledRanking)

            return termRanking

        # Main Logic

        # Check if model already has folders for pairwise performance testing and training results
        if not os.path.exists(f'{self.modelFolder}/Pairwise/Testing'):
            os.mkdir(f'{self.modelFolder}/Pairwise/Testing')
        if not os.path.exists(f'{self.modelFolder}/Pairwise/Training'):
            os.mkdir(f'{self.modelFolder}/Pairwise/Training')

        with torch.no_grad():

            # Load fold's neural network
            network = self.networks[fold]
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            network.to(device)

            # Get fold's testing and training gene split, then create the sets of positive and negative gene pairs (only the negative pair will be used)
            trainGenes, testGenes = self.getDataSplit(fold)
            _ , testNegPairs = self.splitPosNegPairs(testGenes)
            _ , trainNegPairs = self.splitPosNegPairs(trainGenes)

            performance = [] # List to store the performance metrics for each term

            for i, (term, posGenes) in enumerate(self.terms):

                # Evaluate the performance of the fold on its test set and save the results to a csv file
                testRanking = evaluateTermPerformance(i,posGenes,testGenes,testNegPairs)
                if testRanking is not None:
                    testRanking.to_csv(f'{self.modelFolder}/Pairwise/Testing/{term.replace(":","-")}_Fold{fold}.csv',index=False)
                    testAvgPrecision, testAUC = PredictionModel.AveragePrecisionAndAUC(testRanking)   
                else:
                    testAvgPrecision, testAUC = None, None
                
                # Evaluate the performance of the fold on its training set and save the results to a csv file
                trainRanking = evaluateTermPerformance(i,posGenes,trainGenes,trainNegPairs)
                if trainRanking is not None:
                    trainRanking.to_csv(f'{self.modelFolder}/Pairwise/Training/{term.replace(":","-")}_Fold{fold}.csv',index=False)
                    trainAvgPrecision, trainAUC = PredictionModel.AveragePrecisionAndAUC(trainRanking)
                else:
                    trainAvgPrecision, trainAUC = None, None

                # Append the performance metrics to the list
                performance.append((term,testAvgPrecision,trainAvgPrecision,testAUC,trainAUC))

            # Save the performance metrics to a csv file
            pd.DataFrame(performance,columns=['Term','Test Average Precision','Train Average Precision','Test AUC','Train AUC']).to_csv(f'{self.modelFolder}/Pairwise/PerformanceSummary_Fold{fold}.csv',index=False)

    def constructFunctionalRelationshipGraph(self,agnosticBatches=100):
        '''
        This method predicts the GO labels for all pairs of genes in the yeast genome. These predictions are organized into a functional relationship graph where the vertices are genes and the edges are vectors of a pair's confidence values
        Arguements:
            -agnosticBatches - the number of batches to split the agnostic pairs into when predicting their confidence values. If this value is too small, the batches will not fit on your machine's GPU's memory
        '''
        # Determine number of pairs from expressionData object
        pairsLength = self.expressionData.pairsLen 

        # Initialize a tensor to hold the confidence valye vectors for each pair of genes
        confidenceValues = torch.zeros((pairsLength,len(self.terms)),dtype=torch.float32)

        with torch.no_grad():
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            # Feed forward gene pairs from single networks
            for fold in range(self.numFolds):
                network = self.networks[fold].to(device)
                network.eval()

                genes = self.geneFolds[fold]
                pairs = np.array([[geneA,geneB] for geneA in genes for geneB in genes if geneA < geneB],dtype='U10')

                features = self.featuresVector(pairs).to(device)
                outputs = network(features)
                
                # Determine the indices of the pairs in the confidenceValue tensor, then save outputs to those indices
                indices = [self.expressionData.genePairIndex(pair[0],pair[1]) for pair in pairs]
                confidenceValues[indices] = outputs.cpu()

            # Fee forward gene pairs from two networks
            for foldA in range(self.numFolds):
                for foldB in range(foldA+1,self.numFolds):
                    networkA = self.networks[foldA].to(device)
                    networkB = self.networks[foldB].to(device)

                    genesA = self.geneFolds[foldA]
                    genesB = self.geneFolds[foldB]
                    pairs = np.array([[geneA,geneB] for geneA in genesA for geneB in genesB if geneA != geneB],dtype='U10')
                    
                    
                    features = self.featuresVector(pairs).to(device)

                    outputsA = networkA(features)
                    outputsB = networkB(features)
                    outputsAverage = (outputsA + outputsB) / 2.0

                    # Determine indicies and save confidence values
                    indices = [self.expressionData.genePairIndex(pair[0],pair[1]) for pair in pairs]
                    confidenceValues[indices] = outputsAverage.cpu()

            # Feed forward gene pairs whose confidence values are averaged across all networks

            # enumerate all pairs of agnostic genes (first, generate all agnostic-agnostic pairs, then generate all agnostic-gold standard pairs)
            agnostic_agnostic_pairs = np.array([[geneA,geneB] for geneA in self.agnosticGenes for geneB in self.agnosticGenes if geneA < geneB],dtype='U10')
            agnostic_goldStandard_pairs = np.array([[geneA,geneB] for geneA in self.goldStandard for geneB in self.agnosticGenes if geneA != geneB],dtype='U10')
            agnosticPairs = np.concatenate((agnostic_agnostic_pairs,agnostic_goldStandard_pairs),axis=0) 

            # Determine indicies of all agnostic pairs in the confidenceValues tensor
            agnosticIndices = [self.expressionData.genePairIndex(pair[0],pair[1]) for pair in agnosticPairs]

            # Loop through all folds' neural networks and average the output confidence values for each agnostic pair
            for i in range(self.numFolds):

                network = self.networks[i].to(device)
                network.eval()

                # The set of all agnostic pairs is too large to fit into a standard GPU's memory, so process them in 100 batches
                for j in range(agnosticBatches):
                    # Determine the start and end indices of the current batch
                    start = j * len(agnosticPairs) // agnosticBatches
                    end = (j + 1) * len(agnosticPairs) // agnosticBatches if j != agnosticBatches - 1 else len(agnosticPairs) # If last batch, include all remaining pairs

                    agnosticFeatures = self.featuresVector(agnosticPairs[start:end]).to(device)
                    outputs = network(agnosticFeatures)
    
                    confidenceValues[agnosticIndices[start:end]] += outputs.cpu()

            # Average the confidence values for each agnostic pair across all folds, then save the array
            confidenceValues[agnosticIndices] /= self.numFolds
            torch.save(confidenceValues,f'{self.modelFolder}/FunctionalRelationshipGraph.pth')

    def singleGeneRankings(self):
        '''
        This method ranks all gene's in the yeast genome by their predicted functional involvement in each predicted GO term. It does this by taking the weighted average of each gene's connection to all genes annotated a given GO term.
        '''

        # Check if model has a folder for single gene rankings, if not, create it
        if not os.path.exists(f'{self.modelFolder}/SingleGeneRankings'):
            os.mkdir(f'{self.modelFolder}/SingleGeneRankings')

        # Load functional relationship graph from file
        functionalRelationshipGraph = torch.load(f'{self.modelFolder}/FunctionalRelationshipGraph.pth')

        summary = [] # Saves the average precision and AUC for each single gene ranking

        # Iterate through each GO term
        for i, (term, termGenes) in enumerate(self.terms):
            # Create a dictionary to hold the sum of confidence values for each gene
            scoreDict = {gene: 0 for gene in self.allGenes}
            
            # Initialize all pairs of genes where at least on gene in the pair is annotated to the current GO term
            pairs = np.array([[geneA,geneB] for geneA in self.allGenes for geneB in termGenes if (geneA not in termGenes or geneA < geneB)],dtype='U10')
            indices = [self.expressionData.genePairIndex(pair[0],pair[1]) for pair in pairs]
            confidenceValues = functionalRelationshipGraph[indices,i]

            # Loop over all pairs and sum the connection of every gene to genes annotated to the current GO term
            for j in range(len(pairs)):
                if pairs[j][0] in termGenes:
                    scoreDict[pairs[j][1]] += confidenceValues[j].item()
                if pairs[j][1] in termGenes:
                    scoreDict[pairs[j][0]] += confidenceValues[j].item()

            rows = []
            for gene, score in scoreDict.items():
                # print(gene,score)
                label = None
                if gene in termGenes:
                    label = 1
                    averageScore = score / (len(termGenes) - 1)
                elif gene in self.goldStandard:
                    label = -1
                    averageScore = score / len(termGenes)
                else:
                    label = 0
                    averageScore = score / len(termGenes)
                rows.append([gene,averageScore,label])
            
            # Create a DataFrame where each row is a gene, its average confidence value, and its label (1 for positive, -1 for negative, 0 for agnostic), then perform a threshold sweep on this ranking
            scores = pd.DataFrame(rows, columns=['Gene','Confidence Value','Label']) # Sort by score in descending order
            ranking = PredictionModel.thresholdSweep(scores)

            # Save the ranking
            ranking.to_csv(f'{self.modelFolder}/SingleGeneRankings/{term.replace(":", "-")}_SingleGeneRanking.csv', index=False)

            avgPrecision, auc = PredictionModel.AveragePrecisionAndAUC(ranking)
            summary.append((term,avgPrecision,auc))
        
        # Save the summary of all rankings
        pd.DataFrame(summary,columns=['Term','Average Precision','AUC']).to_csv(f'{self.modelFolder}/SingleGeneRankings/Summary.csv',index=False)




        

    #----------------------------------------------------------------------------------#
    #------------------------------- Helper Functions ---------------------------------#   
    #----------------------------------------------------------------------------------#

    def thresholdSweep(table:pd.DataFrame) -> pd.DataFrame:
        '''
        Given a table of samples (genes or gene pairs), confidence values, and labels, this method performs a threshold sweep to generate the total operating characteristic of the predictions
        
        Arguments:
            -table - a pandas DataFrame with the following columns:
                "Confidence Value" - the confidence value of the prediction
                "Label" - the label of the prediction (1 for positive, -1 for negative, 0 for agnostic)
        '''

        # Sort the table by confidence values in descending order
        table.sort_values('Confidence Value',axis=0,ascending=False,inplace=True)


        # Initialize confusion matrix, then intialize function to calculate statistics at each confidence threshold
        truePositives = 0
        falsePositives = 0
        trueNegatives = sum(table['Label'] == -1)
        falseNegatives = sum(table['Label'] == 1)
        
        def calculateStatistics() -> np.ndarray:
            falsePositiveRate = falsePositives / (falsePositives + trueNegatives)
            recall = truePositives / (truePositives + falseNegatives)
            precision = 1 if truePositives + falsePositives == 0 else truePositives / (truePositives + falsePositives)
            return np.array([falsePositiveRate,recall,precision])

        # Array that will track confusion matrix statistics for each sample's confidence threshold
        confusionMatrixStatistics = np.ndarray((len(table),3),dtype=np.float32)

        # Iterate through the samples and update confusion matrix based on label
        for i ,(_, row) in enumerate(table.iterrows()):            
            if row['Label'] == 1:
                truePositives += 1
                falseNegatives -= 1
            elif row['Label'] == -1:
                falsePositives += 1
                trueNegatives -= 1

            confusionMatrixStatistics[i] = calculateStatistics()

        # Save each confusion matrix statistic as a new column in the original table
        table['False Positive Rate'] = confusionMatrixStatistics[:,0]
        table['Recall'] = confusionMatrixStatistics[:,1]
        table['Precision'] = confusionMatrixStatistics[:,2]

        return table
    
    def AveragePrecisionAndAUC(table:pd.DataFrame) -> float:
        '''
        Calculates the auc for ROC curve given a table of gene pairs, confidence values, and labels

        Arguments:
            -table - a pandas DataFrame with columns 'Gene A', 'Gene B', 'Confidence Value', and 'Label'
        '''

        # Sort the table by confidence values in descending order, filter out any agnostic samples as they do not contribute to the AUC or average precision
        table = table.sort_values(by='Confidence Value', ascending=False)
        table = table.loc[table['Label'] != 0]

        # Calculate AUC using recall values
        recall = table['Recall'].values
        auc = np.mean(recall)

        # Sort table by False Positive Rate, then perform convex hull to precision values
        table.sort_values(by='False Positive Rate',inplace=True,ascending=True)
        precision = table['Precision'].values

        for i in range(len(precision)-1,0,-1):
            if precision[i-1] < precision[i]:
                precision[i-1] = precision[i]
        
        # Calculate average precision as the mean of the convex-hulled precision values
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

        table.sort(key=lambda x: x[1]) # Sort by fold number
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

        return np.array(trainGenes,dtype='U10'), np.array(testGenes,dtype='U10')

    def splitPosNegPairs(self,genes:np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        '''
        Given a set of genes, this method returns a tuple of two sets. The first set contains all positive gene pairs (pairs of genes that are co-annotated to at least one GO term) and the second set contains all negative gene pairs (pairs of genes that are not co-annotated to any GO terms and whose smallest co-annotated term has less than 10% of the yeast genome annotated to it).
        '''

        posPairs = []
        negPairs = []
        geneSet = set(genes)

        # Loaf smallest common ancestor array from file, then calculate the threshold for negative pairs given the total number of genes
        smallestCommonAncestors = self.loadSmallestCommonAncestorFile()
        smallestCommonAncestorThreshold = 0.1 * len(self.allGenes) # the maximum number of annotations a pair's smallest common ancestor term can have to be considereed a negative pair

        # Iterate through all pairs of genes
        for row in smallestCommonAncestors:

            # Get each gene's name from its index in expressionData object, and the size of the gene's smallest common ancestor term
            geneA = self.expressionData.geneIndexRev[row[0]]
            geneB = self.expressionData.geneIndexRev[row[1]]
            smallestCommonAncestor = row[2]

            # To be a positive or negative pair, both genes must be in the gene set
            if geneA in geneSet and geneB in geneSet:

                # If the pair is co-annotated to at least one GO term, add it to the positive pairs
                if self.coannotated(geneA,geneB):
                    posPairs.append((geneA,geneB))
                # If the pair is not co-annotated to any GO terms, check if its smallest common ancestor term has less than 10% of the yeast genome annotated to it
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
        
        # Calculate the start and end indices for the batch
        start = i * (batchSize // 2)
        end = (i + 1) * (batchSize // 2)

        # Take the appropriate slices of positive and negative pairs, wrapping around if necessary
        posBatch = np.take(posPairs, range(start,end),mode='wrap',axis=0)
        negBatch = np.take(negPairs, range(start,end),mode='wrap',axis=0)

        return np.concatenate((posBatch,negBatch),axis=0)
    
    def featuresVector(self,batch:np.ndarray) -> torch.Tensor:
        '''
        Given a batch of gene pairs, this method return a 2D vector where each row is the feature vector for a gene pair
        '''

        # Determine the indices of each pair of genes in the expressionData's correlation dictionary array
        indices = torch.IntTensor([self.expressionData.genePairIndex(pair[0],pair[1]) for pair in batch])

        # Fetch the row corresponding to each gene pair's features vector
        features = self.expressionData.correlationDictionary.index_select(0,indices).float()

        return features
    
    def labelsVector(self,batch:np.ndarray) -> torch.Tensor:
        '''
        Given a batch of gene pairs, this method returns a 2D vector where each row is the label vector for a gene pair
        '''
        labels = []

        for pair in batch:
            geneA, geneB = pair

            label = []

            # For each GO term, the label is 1 if the pair of genes are co-annotated to the term, and 0 otherwise
            for _, termGenes in self.terms:
                if geneA in termGenes and geneB in termGenes:
                    label.append(1)
                else:
                    label.append(0)
            labels.append(label)

        return torch.tensor(labels,dtype=torch.float32)
    
    def loadSmallestCommonAncestorFile(self) -> np.ndarray:
        '''
        Loads (or creates) a file containing the size of the smallest common ancestor term for each pair of genes in the yeast genome for a given ontology date
        '''

        packageDir = os.path.dirname(__file__) + '/..'
        smallestCommonAncestorFile = f'{packageDir}/data/SmallestCommonAncestor/{self.ontologyDate}.npy'

        # If file already exists, load it
        if os.path.exists(smallestCommonAncestorFile):
            return np.load(smallestCommonAncestorFile,allow_pickle=True)
    
        # Initialize all pairs of genes in the yeast genome, then calculate the smallest common ancestor for each pair
        print(f'Calculating smallest common ancestor for all pairs of genes with {self.ontologyDate} ontology date. This process may take a while...')
        allPairs = np.array([[self.expressionData.geneIndex[i],self.expressionData.geneIndex[j],0] for i in self.goldStandard for j in self.goldStandard if i < j],dtype=np.int32)
        for i, pair in enumerate(allPairs):
            geneA = self.expressionData.geneIndexRev[pair[0]]
            geneB = self.expressionData.geneIndexRev[pair[1]]
            smallestCommonAncestor = self.parser.smallestCommonAncestor(geneA,geneB)
            allPairs[i][2] = smallestCommonAncestor

        np.save(smallestCommonAncestorFile, allPairs, allow_pickle=True)
        print('Smallest common ancestor file saved to', smallestCommonAncestorFile)
        return allPairs
    
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
            


        

