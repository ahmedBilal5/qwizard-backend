from sense2vec import Sense2Vec
from os import path
s2v = Sense2Vec().from_disk(path.abspath('..\s2v_old'))
