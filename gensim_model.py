from gensim.models import KeyedVectors
from os import path
#abs path: 'D:/Work/Semester_7/FYP_Work/FYP/FYP_Core_logic/Gensim_GloVe_vectors/vectors/vectors.bin'
model_gloVe = KeyedVectors.load(path.abspath('../Gensim_GloVe_vectors/vectors/vectors.bin'))