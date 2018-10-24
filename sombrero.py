import logging
import numpy as np
import scipy.signal as signal
from intervaltree import IntervalTree
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift

import src.utils as utils
from src.document_tree import DocumentTree
from src.bag_of_words import BagOfWords
from src.dot_dict import DotDict


class Detector(object):
    """This class is responsible for detecting the main content in a webpage."""

    def __init__(self, filename: str):
        """filename is the path of a yaml file containing the parameters of
        the algorithm."""
        self.all_params = DotDict()
        self.all_params.load_yaml(filename)
        self.params = self.all_params.content

        # Precomputes the mexican hats at different scales
        self.max_scale = int(2 * self.params.resampling / 3)
        self.hats = np.zeros((self.max_scale, self.params.resampling))
        for scale in range(self.params.min_scale, self.max_scale):
            self.hats[scale, :] = signal.ricker(self.params.resampling, scale)
        self.hats = np.matrix(self.hats)

    def run(self, html: str):
        """This is the main function of the class"""
        # Turns the html into a tree and prepares it.
        root = utils.get_tree(html)
        tree = DocumentTree(root, self.all_params)
        tree.prepareForDetection()

        # Run the Mexican hat analysis.
        self._analyze(tree)

        # Injects the weights into the tree.
        tree.applyContentWeight(self.intervals, self.params.epsilon)

        # Smear weights
        bow = BagOfWords()
        tree.smear(bow)

        return list(filter(lambda x: x[0] != '', tree.getSentences()))

    def _analyze(self, tree):
        profile = tree.getVolumeProfile()
        profile = np.array(list(map(lambda x: x[1], profile)))
        maxV = self.params.max_profile_volume
        profile = (profile < maxV) * profile + (profile >= maxV) * maxV
        self.profile = profile
        self.resampled = signal.resample(profile, self.params.resampling)
        self.alpha = len(self.resampled) / len(self.profile)
        self._mexicanHatResponse(self.resampled)
        self._getMaxima()
        logging.info("Found {0} maxima".format(self.maxima.shape[0]))

    def _mexicanHatResponse(self, sig):
        N = len(sig)
        shifts = np.zeros((N, N))
        max_scale = int(2 * N / 3)
        for j in range(N):
            shifts[:, j] = shift(sig, int(N / 2) - j)
        shifts = np.matrix(shifts)
        self.response = np.array(self.hats[:max_scale, :N] * shifts)

    def _getMaxima(self):
        """Get the maximum response per position and remove the natural response
        of the mexican hat by smoothing the signal much more"""
        maxResponse = gaussian_filter(
            np.max(self.response, axis=0), self.params.smooth_factor)
        base = gaussian_filter(np.max(self.response, axis=0), 50)
        delta = maxResponse - base
        self.delta = delta * (delta > 0)
        self.derivative = self.delta - shift(delta, 1)

        maxima = []
        d = self.derivative  # use an alias to simplify the equations.
        # For each local maxima, keep the position, the scale and the response.
        for i in range(1, len(d) - 2):
            secondOrder = d[i + 1] - d[i]
            if d[i] > 0 and d[i + 1] < 0:
                position = np.argmax(self.response, axis=0)[i]
                maxima.append((i, position, self.delta[i], secondOrder))

        aboveThreshold = list(filter(
            lambda x: x[-1] < -self.params.concavity_threshold, maxima))
        if aboveThreshold:
            maxima = aboveThreshold
        maxima = list(map(lambda x: x[:-1], maxima))

        self.intervals = IntervalTree()
        self.maxima = np.array(maxima)
        if not maxima:
            self.intervals[0:len(self.profile)] = 1.0
            return

        # Normalize the responses and use them as a weight.
        self.maxima[:, 2] /= sum(self.maxima[:, 2])

        # Rescale the scale and position to account for the resampling
        if self.alpha > 0:
            self.maxima[:, 0] /= self.alpha
            self.maxima[:, 1] /= self.alpha

        for m in self.maxima:
            begin = max(0, m[0] - m[1])
            end = min(len(self.profile), m[0] + m[1])
            self.intervals[begin:end] = m[2]
