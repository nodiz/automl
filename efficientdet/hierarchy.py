import pickle
from os.path import join
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

leaves_hierachy = None


@dataclass
class Node:
    """Class for keeping track of a leaf node"""
    id: str
    story: OrderedDict
    denominator: float
    data: dict = None
    name: str = ''
    vector: list = None

    def getDistance(self, other) -> float:
        for c in self.story:  # ordered
            if c in other.story:  # complexity O(1)
                return self.story[c]


def getHierarchy(basePath=''):
    global leaves_hierachy
    if leaves_hierachy is None:
        leaves_hierachy = pickle.load(open(join(basePath, "hierarchy.pkl"), "rb"))
    return leaves_hierachy


node0 = Node('0', OrderedDict({}), 1, vector=np.ones[101])


def getVector(n):
    global leaves_hierachy, node0
    if n == 0:
        return node0
    else:
        return leaves_hierachy[n-1].vector


def constructBigMatrix(datas):
    bigTensor = np.zeros(datas.shape + (101,))
    for i, x1 in enumerate(datas):
        for j, x2 in enumerate(x1):
            for k, x3 in enumerate(x2):
                for m, x4 in enumerate(x3):
                    bigTensor[i, j, k, m] = getVector(datas[i, j, k, m])
    return bigTensor
