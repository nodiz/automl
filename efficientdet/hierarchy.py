import pickle
from os.path import join
import numpy as np
import tensorflow as tf

leaves_hierachy = None

def getHierarchy(basePath='config/'):
    global leaves_hierachy
    if leaves_hierachy is None:
        leaves_hierachy = pickle.load(open(join(basePath, "hierarchy_vects.pkl"), "rb"))
    return tf.constant(leaves_hierachy)


if __name__ == "__main__":
    leaves = getHierarchy()
    print(type(leaves))
    print(leaves.shape)

    a = tf.Variable(np.random.rand(8, 80, 80, 9, 101))
    b = a@leaves
    print(b.shape)