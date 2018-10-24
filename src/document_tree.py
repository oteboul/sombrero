from collections import defaultdict

import src.utils as utils
from src.bag_of_words import BagOfWords


class DocumentTree(object):
    """This represent a document as a tree for content detection."""

    _BLOCK_ELEMENTS = set([
        "div", "h1", "h2", "h3", "h4", "p", 'form', 'ul', 'ol', 'table',
        "title", "body", "img", "header",
    ])
    _BANNED_ELEMENTS = set(["script", "meta", "link", "style"])

    def __init__(self, node, params, kls="-", parent=None):
        """Node is an HTML element."""
        self.params = params
        self.parent = parent

        # The description of the node.
        self.tag = node.tag
        self.attrib = node.attrib
        currKls = node.get("class")
        if not currKls:
            currKls = kls
        self.kls = currKls

        # Different weights for this node.
        self.heaps = 0.0
        self.heapsWeight = 1.0
        self.htmlWeight = 1.0
        self.contentWeight = 1.0
        self.weight = 1.0

        # The content of the node
        self.text = node.text if node.text else ""
        self.tail = node.tail if node.tail else ""

        self.bow = BagOfWords()
        self.bow.accountFor(self.text)
        self.bow.accountFor(self.tail)
        self.raw = BagOfWords()
        self.raw.copy(self.bow)

        # The children of the node.
        self.children = []
        for child in node:
            if child.tag not in self._BANNED_ELEMENTS:
                self.children.append(
                    DocumentTree(child, self.params, self.kls, self))

    def hasAncestor(self, tag):
        if self.parent is None:
            return False

        if self.parent.tag == tag:
            return True

        return self.parent.hasAncestor(tag)

    def prepareForDetection(self):
        """Before analyzing the tree by a detector, it must be weighted and
        flatten"""
        heapsWeights = self.computeTagWeights()
        self.setHeapsWeights(heapsWeights)
        self.flatten()

    def textPerTag(self):
        """get all the text per (tag, kls)"""
        def _aux(node, tags):
            key = (node.tag, node.kls)
            for txt in [node.text, node.tail]:
                if txt:
                    tags[key].append(txt)
            for child in node.children:
                _aux(child, tags)

        tags = defaultdict(list)
        _aux(self, tags)
        return tags

    def computeTagWeights(self):
        """For each (tag, kls), compute a weight based on Heap's law,
        by aggregating per (tag, kls)"""
        tags = self.textPerTag()
        result = {}
        for tag, texts in tags.items():
            bow = BagOfWords()
            for text in texts:
                bow.accountFor(text)
            weight = bow.computeHeapsWeight(self.params.heaps)
            weight *= bow.computeNumericWeight(self.params.numeric)
            result[tag] = weight
        return result

    def setHeapsWeights(self, weights):
        self.heapsWeight = weights.get((self.tag, self.kls), 1.0)
        for child in self.children:
            child.setHeapsWeights(weights)

    def flatten(self):
        texts = []
        hasBlock = False
        toRemove = []
        self.bow *= self.heapsWeight

        for childIndex, child in enumerate(self.children):
            child.flatten()
            if child.tag not in self._BLOCK_ELEMENTS:
                curr = ""
                if child.text:
                    curr += child.text.strip()

                self.bow += child.bow
                self.raw += child.raw
                texts.append(utils.remove_duplicate_spaces(curr))
                toRemove.append(childIndex)
            else:
                hasBlock = True

        for i in reversed(toRemove):
            self.children.pop(i)

        childrenText = utils.remove_duplicate_spaces(" ".join(texts))
        if self.text:
            self.text += childrenText
        else:
            self.text = childrenText
        if self.tail:
            self.text += self.tail

        if hasBlock and self.tag not in self._BLOCK_ELEMENTS:
            self.tag = "div"

    def applyContentWeight(self, intervals, epsilon):
        """Given a list of center, scale and weights, we apply"""
        def _aux(node, visited=[]):
            node.position = len(visited)
            node.endContent = intervals.end()
            node.contentWeight = max(
                sum(map(lambda x: x.data, intervals[node.position])), epsilon)
            visited.append(node)
            for child in node.children:
                _aux(child, visited)
        visited = []
        _aux(self, visited)

    def smear(self, bow):
        self.htmlWeight = self.params.html.get(self.tag, 1.0)

        # Titles should be head title...
        if self.tag == 'title':
            if self.hasAncestor('body'):
                self.htmlWeight = 1.0

        # h1 only before the content, not after.
        self.weight = self.htmlWeight
        if self.weight > 1 and self.position > self.endContent:
            self.weight = 1.0

        if self.weight == 1.0:
            self.weight = self.contentWeight

        bow += self.bow * self.weight
        for child in self.children:
            child.smear(bow)
            self.raw += child.raw

        return bow

    def getVolumeProfile(self):
        def _aux(node, result):
            result.append(
                ("{0}:{1}".format(node.tag, node.kls), node.bow.volume))
            for child in node.children:
                _aux(child, result)
            return result
        result = []
        return _aux(self, result)

    def getSentences(self, threshold=0.01):
        def aux(node, result):
            if node.weight > threshold:
                result.append((node.text.strip(), node.weight))
            for child in node.children:
                aux(child, result)
        result = []
        aux(self, result)
        return result
