from Ontology import Ontology

class GOParser():
    '''
    
    '''
    
    def __init__(self,ontoDate,ontoFile='',ontoAnnos='',slimFile=''):
        
        
        if ontoDate == '2007' or ontoDate == 'original':
            file = './obopy/gene_ontology_2007_Jan.obo'
            annos = './obopy/gene_association.sgd.20070415/gene_association.sgd'
        elif ontoDate == '2009':
            file = 'obopy/gene_ontology_2009_Jan.obo'
            annos = './obopy/sgd_2009_Jan_unzip.gaf'
        elif ontoDate is not None:
            file = './obopy/go-basic.obo'
            annos = './obopy/sgd.gaf'

        if ontoFile != '':
            file = ontoFile
        if ontoAnnos != '':
            annos = ontoAnnos

        self.onto = Ontology(file,annos,loadLocal=True)
        self.assignChildren(self.onto)
        self.assginYORF(self.onto)
        for id, gene in self.onto.genes.items():
            gene.annotateTerms()

        self.numGenes = len(self.onto.genes)


        if slimFile != '':
            sFile = slimFile
        else:
            sFile = './obopy/goslim_yeast.obo'

        self.slim = Ontology(sFile,annos,loadLocal=True)
        self.assignChildren(self.slim)
        self.assginYORF(self.slim)


    def assignChildren(self,ontology):
        for id, term in ontology.terms.items():
            for parent in term.parents():
                parent.children.add(term)
            

    def assginYORF(self,ontology: Ontology):
        for id, gene in ontology.genes.items():
            for alias in gene.aliases:
                if GOParser.isYORF(alias):
                    ontology.yorfs[alias] = gene

    def getGenes(self,term,direct=False):
        ontoTerm = self.onto.terms[term]
        if direct:
            genes = ontoTerm.directAnnos()
        else:
            genes = ontoTerm.allAnnos()

        yorfList = set()
        #Loop over all genes in sets
        for gene in genes:
            #Loop over all names for that gene
            for name in gene.aliases:
                #If name matches character requirements, and adds to yorfList
                if GOParser.isYORF(name):
                    yorfList.add(name)
        return yorfList
    
    def isYORF(name):
        return (len(name) >= 7 and name[0] == 'Y' and (name[2] == 'L' or name[2] == 'R') and name[3:5].isdigit() and (name[6] == 'W' or name[6] == 'C'))
                    


    def getSlimLeaves(self,cutoff=10,roots='b',childDepth=0,onlyLeaves=True):
        bioProc = self.slim.terms['GO:0008150']
        cellComp = self.slim.terms['GO:0005575']
        molFunc = self.slim.terms['GO:0003674']
        def rootFilter(term):
            ancestors = term.ancestors()
            return (bioProc in ancestors and 'b' in roots) or (cellComp in ancestors and 'c' in roots) or (molFunc in ancestors and 'm' in roots)
        

        leaves = []
        for id, term in self.slim.terms.items():
            if id in self.onto.terms and rootFilter(term):
                
                termGenes = self.getGenes(id)

                
                if len(termGenes) >= cutoff and (len(self.slim.terms[id].children) == 0 or (not onlyLeaves)):
                    leaves.append((id,termGenes))
        return leaves
    
    def smallestCommonAncestor(self,geneA,geneB):
        A_terms = self.onto.yorfs[geneA].allTerms
        B_terms = self.onto.yorfs[geneB].allTerms

        shared = A_terms & B_terms
        # for term in shared:
        #     print(term.name)
        #     print(len(term.allAnnos()))
        
        shared_geneNums = list(map(lambda term: len(term.allAnnos()),shared))
        return min(shared_geneNums)





            

