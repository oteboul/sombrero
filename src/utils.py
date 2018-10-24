import lxml
import lxml.html
import string
import sys

from functools import partial
from math import exp, log


_MAX_EXP_ARGUMENT = log(sys.float_info.max)


def get_tree(html: str, remove_comments=True):
    tree = lxml.html.fromstring(html)
    if remove_comments:
        lxml.etree.strip_tags(tree, lxml.etree.Comment)
    return tree


def remove_duplicate_spaces(text: str) -> str:
    space = " "
    for key in ["\n", "\t"]:
        text = text.replace(key, space)
    return " ".join(filter(None, text.strip().split(space)))


def remove_punctuation(text: str, toSpace=[], keep=set(), extras=[]) -> str:
    mapping = {}
    for key in string.punctuation:
        if key not in keep:
            mapping[key] = None
    for key in ["\n", "\t"]:
        mapping[key] = None
    for key in extras:
        mapping[key] = None
    mapping["'"] = "' "
    for char in toSpace:
        mapping[char] = " "
    translator = str.maketrans(mapping)
    return text.translate(translator)


def get_tokens(text: str, toSpace=[], keep=set(), extras=[]) -> list:
    clean = remove_punctuation(text, toSpace, keep, extras)
    normalized = \
        filter(lambda x: x != "*", map(lambda x: x.lower(), clean.split(" ")))
    return " ".join(filter(None, normalized)).split(" ")


def sigmoid(x, s, h):
    arg = -s * (x - h)
    if arg > _MAX_EXP_ARGUMENT:
        return 0.0

    return 1.0 / (1.0 + exp(arg))


def bounded_sigmoid(x, start, half, end, slope):
    if x > end:
        return 1.0

    if x < start:
        return 0.0

    sig = partial(sigmoid, s=slope, h=half)
    y0 = sig(start)
    y1 = sig(end)
    y = sig(x)
    return (y - y0) / (y1 - y0)


def sigmoid_from_params(x, params):
    value = 0.0
    if "start" in params and "end" in params:
        value = bounded_sigmoid(
            x, params.start, params.half, params.end, params.slope)
    else:
        value = sigmoid(x, params.slope, params.half)

    if params.decreasing:
        value = 1.0 - value

    return params.min + (params.max - params.min) * value
