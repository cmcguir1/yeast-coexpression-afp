from Ontology import Ontology
import os
from Gene import Gene

class GOParser():
    '''
    This class parses and modify Ontology objects to create the slim leaf terms object used in extensively in the PredictionModel class. It also contains a method to get all gene yorfs annotated to a particular term.
    '''

    def __init__(self,ontologyDate,slimFile='goslim_yeast.obo'):
        '''
        Initialization of GOParser object.

        Arguments:
            -ontologyDate - date from which the GO obo ontology file and sgd annotations file will be pulled from. date should be specified as a string in format 'YYYY-MM-DD'.
            -slimFile - file path to GO slim file, which contains a pruned version of the Gene Ontology with terms hand currated by experts to be informative for directing labratory experiments
        '''
        packageDir = os.path.dirname(__file__) + '/../..'
        if os.path.exists(f'{packageDir}/data/GO/{ontologyDate}/go-basic.obo'):
            ontologyFile = f'{packageDir}/data/GO/{ontologyDate}/go-basic.obo'
        else:
            ontologyFile = f'{packageDir}/data/GO/{ontologyDate}/gene_ontology.obo'

        annotationsFile = f'{packageDir}/data/GO/{ontologyDate}/sgd.gaf'
        slimFile = f'{packageDir}/data/GO/{slimFile}'
        
        # Create onotology and slim ontology objects
        self.ontology = Ontology(ontologyFile,annotationsFile,loadLocal=True)
        self.slim = Ontology(slimFile,annotationsFile,loadLocal=True)
        

    def getGenes(self,term:str) -> set[str]:
        '''
        This gene loops through all of the genes annotated to a term, then returns a set of all of the gene names that are YORFs
        '''
        ontoTerm = self.ontology.terms[term]
        termGenes = ontoTerm.allAnnos()

        yorfs = set()
        for gene in termGenes:
            for name in gene.aliases:
                if GOParser.isYORF(name):
                    yorfs.add(name)
        return yorfs
    

    def isYORF(name: str) -> bool:
        ''' Predicate that returns true to name is a YORF (Yeast Open Reading Frame)'''
        return (len(name) >= 7 and name[0] == 'Y' and (name[2] == 'L' or name[2] == 'R') 
                and name[3:5].isdigit() and (name[6] == 'W' or name[6] == 'C'))


    def getSlimLeaves(self,cutoff:int=10,roots:str='b',onlyLeaves:bool=True) -> list[tuple[str,set[str]]]:
        '''
        This method returns a list of tuples where the first element is the a GO term ID for a GO term in the GO slim and the second element is the set of all genes annotated to that GO term.

        Arguments:
            -cutoff - minimum number of annotations a term must have to be included in returned slim
            -roots - a string single character flags specifying from which sub-ontologies terms are pulled. The flags are as follows:
                b - biological process
                c - cellular component
                m - molecular function
            -onlyLeaves - boolean flag specifying whether or not to only return leaf terms in the slim
        '''

        # Intialize root terms
        bioProc = self.slim.terms['GO:0008150']
        cellComp = self.slim.terms['GO:0005575']
        molFunc = self.slim.terms['GO:0003674']

        def rootFilter(term:str) -> bool:
            ''' Predicate that returns true if term belongs to sub-ontology specified by roots '''
            ancestors = term.ancestors()
            return ((bioProc in ancestors and 'b' in roots) 
                    or (cellComp in ancestors and 'c' in roots) 
                    or (molFunc in ancestors and 'm' in roots))

        leaves = []
        for id, term in self.slim.terms.items():
            if id in self.ontology.terms and rootFilter(term):
                termGenes = self.getGenes(id)
                
                # If term has more annotations than cutoff and is a leaf term (has no children in slim), add term to list of returned terms
                if (len(termGenes) >= cutoff 
                    and (len(self.slim.terms[id].children) == 0 or (not onlyLeaves))):
                    leaves.append((id,termGenes))

        return leaves
    

    def smallestCommonAncestor(self,geneA:str,geneB:str) -> int:
        '''
        This method returns the number of annotations of the smallest GO term for which geneA and geneB are co-annotated to. This method is used in the PredictionModel to remove gene pairs that share a a similar broad function, but are not labled as positive.
        '''

        # Get each gene's annotated terms, the take their intersection to get the set of co-annotated terms
        A_terms = self.onto.yorfs[geneA].allTerms
        B_terms = self.onto.yorfs[geneB].allTerms
        coannotations = A_terms & B_terms
        
        # mapping of shared GO terms to number of annotations to that GO term
        shared_geneNums = list(map(lambda term: len(term.allAnnos()),coannotations))
        return min(shared_geneNums)
    
    def allGenes(self) -> set[str]:
        '''
        This method returns a list of all genes contained within the parser's ontology
        '''
        return self.ontology.yorfs.keys()



            

