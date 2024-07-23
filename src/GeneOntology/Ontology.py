from OntoTerm import OntoTerm
from Gene import Gene
import urllib.request

class Ontology:
    def __init__(self,oboFileName=None,annoFileName=None,
                 loadLocal=False,bypassOrigCheck=False,
                 annotateTerms=False):
        self.originalCheck = not bypassOrigCheck
        
        #Fields
        self.terms = {} #Term.uid --> Term
        self.roots = [] #Root terms
        self.genes = {} #Gene.uid --> Gene

        self.yorfs = {} #YORF -> Gene
        
        self.aliasGenes = {} #GeneAlias --> Gene.uid

        #Load files if able
        if oboFileName is not None:
            self.loadOboFile(oboFileName, loadLocal)
            if annoFileName is not None:
                self.loadAnnoFile(annoFileName, loadLocal)

        self.assignChildren()
        self.assginYORF()
        if annotateTerms:
            for _, gene in self.genes.items():
                gene.annotateTerms()

    # I wrote this method
    def assignChildren(self):
        for id, term in self.terms.items():
            for parent in term.parents():
                parent.children.add(term)

    # I wrote this method
    def assginYORF(self):
        for id, gene in self.genes.items():
            for alias in gene.aliases:
                if Ontology.isYORF(alias):
                    self.yorfs[alias] = gene

    # I wrote this method. It checks if a gene name is a YORF
    def isYORF(name):
        return (len(name) >= 7 and name[0] == 'Y' and (name[2] == 'L' or name[2] == 'R') and name[3:5].isdigit() and (name[6] == 'W' or name[6] == 'C'))


    def loadOboFile(self, oboFileName, loadLocal=False):
        if loadLocal:
            fin = open(oboFileName,"r")
        else:
            fin = urllib.request.urlopen(oboFileName)
        #Parsing helper variables
        interm = False
        uid = ""
        name = ""
        defn = ""
        is_a = set()
        part_of = set()
        regulates = set()
        #Parse the OBO file format
        for line in fin:
            if not loadLocal:
                line = line.decode()
            line = line.strip()
            if line == "":
                if interm: #If blank line and was reading in a term, add it
                    if uid not in self.terms:
                        self.terms[uid] = OntoTerm(uid,name,defn,is_a,part_of,regulates)
                    else:
                        term = self.terms[uid]
                        term.name = name
                        term.defn = defn
                        term.is_a = is_a
                        term.part_of = part_of
                        term.regulates = regulates
                    if len(self.terms[uid].parents()) == 0:
                        self.roots.append(self.terms[uid])
                interm = False
            elif line == "[Term]":
                #Reset everything for this term
                interm = True
                uid = ""
                name = ""
                defn = ""
                is_a = set()
                part_of = set()
                regulates = set()
            elif interm:
                if line[0:3] == "id:":
                    uid = line[4:]
                    # self.terms[uid] = OntoTerm(uid) # My addition
                elif line[0:5] == "name:":
                    name = line[6:]
                elif line[0:4] == "def:":
                    defn = line[5:]
                elif line[0:12] == "is_obsolete:":
                    interm = False
                elif line[0:5] == "is_a:":
                    termid = line[6:16]
                    if termid not in self.terms:
                        self.terms[termid] = OntoTerm(termid)
                    # self.terms[termid].children.add(self.terms[uid]) # My addition, this should allow for keeping track of terms children easily
                    is_a.add(self.terms[termid])
                elif line[0:21] == "relationship: part_of":
                    termid = line[22:32]
                    if termid not in self.terms:
                        self.terms[termid] = OntoTerm(termid) 
                    # self.terms[termid].children.add(self.terms[uid]) # My addition, this should allow for keeping track of terms children easily
                    part_of.add(self.terms[termid])
                elif line[0:23] == "relationship: regulates":
                    termid = line[24:34]
                    if termid not in self.terms:
                        self.terms[termid] = OntoTerm(termid)
                    # self.terms[termid].children.add(self.terms[uid]) # My addition, this should allow for keeping track of terms children easily
                    regulates.add(self.terms[termid])
                elif line[0:34] == "relationship: positively_regulates" or line[0:34] == "relationship: negatively_regulates":
                    termid = line[35:45]
                    if termid not in self.terms:
                        self.terms[termid] = OntoTerm(termid)
                    # self.terms[termid].children.add(self.terms[uid]) # My addition, this should allow for keeping track of terms children easily
                    regulates.add(self.terms[termid])
        fin.close()

    def loadAnnoFile(self, annoFileName, loadLocal=False):
        if loadLocal:
            fin = open(annoFileName, "r",encoding='cp437')
        else:
            fin = urllib.request.urlopen(annoFileName)
        for line in fin:
            #Apparently in this version of python, strings are already decoded
            #line = line.decode()
            line = line
            if line[0] != "!":
                line = line.strip()
                parts = line.split("\t")


                #print(parts)


                #Get gene instance (Create it if needed)
                if len(parts) < 3:

                    
                    #print("Short line: " + line)

                    pass
                if self.originalCheck and parts[2] not in self.genes :
                    newGene = Gene(parts[1], parts[2], parts[9], set(parts[10].split("|")))
                    self.genes[parts[2]] = newGene

                    #Since this is a new gene, add its aliases
                    #Keep the symbol as the primary mapping (e.g. don't let other things map to a valid symbol)
                    if newGene.symbol in self.aliasGenes and self.aliasGenes[newGene.symbol] != newGene.symbol:
                        #print(f"WARNING: Alias conflict: Mapping {newGene.symbol} to {newGene.symbol} but already mapped to {self.aliasGenes[newGene.symbol]}")
                        newGene.symbol
                    else:
                        self.aliasGenes[newGene.symbol] = newGene.symbol
                    #The UID should also map only to its symbol (e.g. UIDs aren't allowed to map to other things)
                    if newGene.uid in self.aliasGenes and self.aliasGenes[newGene.uid] != newGene.symbol:
                        #print(f"WARNING: Alias conflict: Mapping {newGene.uid} to {newGene.symbol} but already mapped to {self.aliasGenes[newGene.uid]}")
                        newGene.symbol
                    else:
                        self.aliasGenes[newGene.uid]    = newGene.symbol
                        
                    #Other aliases, let's just ignore them for now; but this is where some real work should happen
                    #TODO: Better dealing with unclear aliases
                    for alias in newGene.aliases:
                        if alias in self.aliasGenes and self.aliasGenes[alias] != newGene.symbol:
                            #print(f"WARNING: Alias conflict: Mapping {alias} to {newGene.symbol} but already mapped to {self.aliasGenes[alias]}")
                            newGene.symbol
                        #self.aliasGenes[alias]      = newGene.symbol
                    
                g = self.genes[parts[2]]
                #Annotate the gene to Terms
                if "NOT" not in parts[3]: #Eliminate negative annotations
                    if parts[4] not in self.terms:

                        
                        #print("ERROR: " + parts[4] + "not found in Ontology")
                        pass

                    else:
                        term = self.terms[parts[4]]
                        term.annotate(g, parts[6])

    def enrichByGene(self, qGenes):
        termsToCheck = set()
        pvalResults = {}
        for gene in qGenes:
            termsToCheck = termsToCheck.union(gene.terms())
        for term in termsToCheck:
            pvalResults[term] = term.hyperGeomEnrich(qGenes, len(self.roots[0].allAnnos()))
        return pvalResults

    def enrichByGeneNames(self, qGeneNames):
        qGenes = set()
        print(qGeneNames)
        for name in qGeneNames:
           print(name)
           if name in self.genes:
               qGenes.add(self.genes[name])
        print("Hello!")
        print(qGeneNames)
        print(qGenes)
        if len(qGenes) > 0:
           print(len(qGenes))
           print(qGenes)
           return self.enrichByGene(qGenes)
        else:
            return {}

"""
go = Ontology("go.obo","gene_association.mgi")
for root in go.roots:
    print("*" + root.name)
    print("**" + str(len(root.directAnnos())))
    print("**" + str(len(root.allAnnos())))
print(len(go.genes))

pvals = go.enrichByGeneNames(["Notch1","Notch2","Notch3"])
for term in pvals:
    print(term.uid + "\t" + pvals[term] + "\t" + term.name)
"""