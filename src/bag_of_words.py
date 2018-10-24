from collections import defaultdict

import src.utils as utils


class BagOfWords(object):
    """This class represents a bag-of-words and adds the possibilities to
    do simple mathemetical operation with them such as sum or scalar
    multiplication"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.bow = defaultdict(float)
        self.volume = 0.0
        # Volume of numeric terms
        self.numeric = 0.0
        # Number of times we added some text to this bag of word.
        self.numTexts = 0

    def copy(self, other):
        # Maybe deepcopy could be useful here.
        self.reset()
        for key in other.bow:
            self.bow[key] = other.bow[key]
        self.volume = other.volume
        self.numeric = other.numeric
        self.numTexts = other.numTexts

    def __add__(self, other):
        result = BagOfWords()
        if isinstance(other, BagOfWords):
            keys = set(self.bow.keys()) | set(other.bow.keys())
            for key in keys:
                result.bow[key] = self.bow[key] + other.bow[key]
            result.volume = self.volume + other.volume
            result.numeric = self.numeric + other.numeric
            result.numTexts = self.numTexts + other.numTexts
        else:
            result.volume = self.volume
            for key in self.bow:
                result.bow[key] = self.bow[key] + other
                if ' ' not in key:
                    result.volume += other
                    result.numeric += float(other.isnumeric())
        return result

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        for key, count in other.bow.items():
            self.bow[key] += count
        self.volume += other.volume
        self.numeric += other.numeric
        self.numTexts += other.numTexts
        return self

    def __mul__(self, scalar):
        result = BagOfWords()
        for k in self.bow:
            result.bow[k] = scalar * self.bow[k]
        result.volume = scalar * self.volume
        result.numeric = scalar * self.numeric
        return result

    def __rmul__(self, other):
        return self * other

    def __imul__(self, scalar):
        for k in self.bow:
            self.bow[k] *= scalar
        self.volume *= scalar
        self.numeric *= scalar
        return self

    def __pow__(self, p):
        result = BagOfWords()
        for k in self.bow:
            result.bow[k] = self.bow[k] ** p
        result.computeVolume()
        return result

    def __getitem__(self, key):
        return self.bow[key]

    def __contains__(self, key):
        return key in self.bow

    def computeVolume(self):
        self.volume = 0.0
        self.numeric = 0.0
        for key, count in self.bow.items():
            if ' ' not in key:
                self.volume += count
                self.numeric += count * key.isnumeric()

    def addWord(self, word, isUnigram=True):
        self.bow[word] += 1.0
        if isUnigram:
            self.volume += 1.0
            self.numeric += word.isnumeric()

    def accountFor(self, text):
        tokens = utils.get_tokens(text, toSpace=["-", "«", "»", "\xa0"])
        if not tokens:
            return

        self.numTexts += 1

        # Counts the terms up to trigrams.
        numTokens = len(tokens)
        left = None
        for i in range(numTokens):
            curr = tokens[i]
            if not curr:
                continue

            right = None
            if i < numTokens - 1:
                right = tokens[i + 1]

            self.addWord(curr, isUnigram=True)
            if right:
                bigram = " ".join((curr, right))
                self.addWord(bigram, isUnigram=False)
            if left and right:
                trigram = " ".join((left, curr, right))
                self.addWord(trigram, isUnigram=False)
            left = curr

    def computeHeapsWeight(self, params):
        numUnigrams = len(list(filter(lambda x: " " not in x, self.bow.keys())))
        if self.volume < params.min_volume or numUnigrams == 0 or \
                self.numTexts < 4:
            return 1.0

        value = 100 * self.volume ** params.power / numUnigrams
        return utils.sigmoid_from_params(value, params.sigmoid)

    def computeNumericWeight(self, params):
        if not self.volume:
            return 0.0

        value = self.numeric / self.volume
        return utils.sigmoid_from_params(value, params)
