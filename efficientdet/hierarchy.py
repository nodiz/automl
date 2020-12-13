import pickle
import numpy as np
from os.path import join
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

leaves_hierachy = None

from make_hierarchy import Node

def getHierarchy(basePath='config/'):
    global leaves_hierachy
    if leaves_hierachy is None:
        leaves_hierachy = pickle.load(open(join(basePath, "hierarchy.pkl"), "rb"))
    return leaves_hierachy


node0 = Node('0', OrderedDict({}), 1, vector=np.ones(101))


def getVector(n):
    global leaves_hierachy, node0
    if n == 0:
        return node0
    else:
        return leaves_hierachy[n-1].vector

@tf.function
def constructBigMatrix(datas):
    datas_size = datas.shape

    @tf.function
    def expand_values(x):
        return tf.cast(tf.constant(getVector(x)), dtype=x.dtype)

    datas = tf.reshape(datas,(-1,1))

    datas  = tf.map_fn(expand_values, datas, back_prop=True, parallel_iterations=10, infer_shape=True)

    datas = tf.reshape(datas, datas_size+(101,))

    return datas



if __name__ == "__main__":
    leaves = getHierarchy()
    print(getVector(1))
    a = tf.Variable(np.random.rand(8,40,40,9))
    b = constructBigMatrix(a)
    print(b.shape)
