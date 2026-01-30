class TTRLClusterCounter:
    def __init__(self, equivalence_func, iterable=None):
        if not callable(equivalence_func):
            raise TypeError("equivalence_func must be a callable function")
            
        self.is_equivalent = equivalence_func
        self._clusters = {}
        
        if iterable:
            self.update(iterable)

    def add(self, item):
        for representative in self._clusters.keys():
            if self.is_equivalent(item, representative):
                self._clusters[representative].append(item)
                return
        self._clusters[item] = [item]

    def update(self, iterable):
        for item in iterable:
            self.add(item)
            
    @property
    def counts(self):
        return {rep: len(members) for rep, members in self._clusters.items()}

    def most_common(self, n=None):
        sorted_items = sorted(self.counts.items(), key=lambda item: item[1], reverse=True)
        if n is None:
            return sorted_items
        return sorted_items[:n]
        
    def __len__(self):
        return len(self._clusters)

    def __str__(self):
        counts_str = {str(k): v for k, v in self.counts.items()}
        return f"{self.__class__.__name__}({counts_str})"

    def __repr__(self):
        return self.__str__()
    
    def items(self):
        return self._clusters.items()