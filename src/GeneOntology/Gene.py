from OntoTerm import OntoTerm

class Gene:
    def __init__(self, _uid=None, _symb=None, _name=None, _alias=set()):
        self.uid = _uid
        self.symbol = _symb
        self.name = _name
        self.aliases = _alias
        self.direct = {} #Code --> set(OntoTerm)
        self.annos = {} #Code --> set(OntoTerm)
        self.children = set()
        self.allTerms = set()

    def __str__(self):
        return(self.symbol)
    def __repr__(self):
        return (str(self))

    def toJSON_summary(self):
        result = {}
        result["uid"] = self.uid
        result["symbol"] = self.symbol
        result["name"] = self.name
        return(result)

    def toJSON_complete(self):
        result = self.toJSON_summary()
        result["alias"] = [x for x in self.aliases]
        #result["direct"] = [x.uid for x in self.direct]
        result["annos"] = [x.uid for x in self.allAnnos()]
        return(result)

    def terms(self, exclude=None):
        result = set()
        for code in self.annos:
            result = result.union(self.annos[code])
        return result

    def directAnnotate(self, term, code):
        if code not in self.direct:
            self.direct[code] = set()
        self.direct[code].add(term)

    def addAnnos(self, term, code):
        if code not in self.annos:
            self.annos[code] = set()
        self.annos[code].add(term)

    def allAnnos(self):
        result = set()
        for code in self.annos:
            result = result.union(self.annos[code])
        return(result)

    def allNames(self):
        result = set()
        result.add(self.uid)
        result.add(self.symbol)
        for alias in self.aliases:
            result.add(alias)
        return result
    
    def annotateTerms(self):
        self.allTerms = self.terms()