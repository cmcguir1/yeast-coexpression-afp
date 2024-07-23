import scipy.stats as stats

class OntoTerm:
    uid = ""
    name = ""
    defn = ""
    is_a = set() #Set of OntoTerms that are direct parents
    part_of = set()
    regulates = set()

    def __init__(self, _uid="", _name="", _defn="", _is_a=set(), _part_of=set(), _regulates=set()):
        self.uid  = _uid
        self.name = _name
        self.defn = _defn
        self.is_a = _is_a
        self.part_of = _part_of
        self.regulates = _regulates
        self.direct = {} #Code --> Gene
        self.annos = {} #Code --> Gene
        self.children = set()
        
        

    def __str__(self):
        return(self.uid)
    def __repr__(self):
        return(self.uid + " (" + self.name + ")")

    def parents(self):
        return(self.is_a.union(self.part_of).union(self.regulates))
    
    def descendants(self):
        allDescend = self.children.union(set())
        for term in allDescend:
            allDescend = allDescend.union(term.descendants())
        return allDescend

    
    def ancestors(self):
        allAncestors = self.parents()
        for term in self.parents():
            allAncestors = allAncestors.union(term.ancestors())
        return allAncestors

    def directAnnos(self):
        result = set()
        for code in self.direct:
            result = result.union(self.direct[code])
        return result

    def allAnnos(self):
        result = set()
        for code in self.annos:
            result = result.union(self.annos[code])
        return result

    def annotate(self, gene, code):
        #Add gene to term
        if code not in self.direct:
            self.direct[code] = set()
        self.direct[code].add(gene)
        #Add term to gene
        gene.directAnnotate(self,code)
        #Recursively annotate up
        self.addAnnos(gene, code)

    def addAnnos(self, gene, code):
        #Add gene to the annotations of this term
        if code not in self.annos:
            self.annos[code] = set()
        if gene not in self.annos[code]:
            if(self.uid == "GO:1904382"):
                #print(str(gene) + "\t" + code)
                pass
            self.annos[code].add(gene)
            #Add term to gene
            gene.addAnnos(self,code)
            #Recursively annotate the gene upward
            for parent in self.parents():
               # if gene.symbol == "MNL1":
                    #print("MNL1: " + str(self) + " -> " + str(parent))
                parent.addAnnos(gene, code)

    def hyperGeomEnrich(self, genes, totalGenes):
        termGenes = self.allAnnos()
        hgd = stats.hypergeom(totalGenes, len(termGenes), len(genes))
        overlap = termGenes.intersection(genes)
        print(self.uid + "\t" + self.name + "\t" + str(totalGenes) + "\t" + str(len(termGenes)) + "\t" + str(len(genes)) + "\t" + str(len(overlap)))
        p = 0
        for x in range(len(overlap),len(genes)+1):
            p += hgd.pmf(x)
        return p
