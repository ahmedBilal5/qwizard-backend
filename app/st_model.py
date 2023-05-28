from sentence_transformers import SentenceTransformer, util
from os import path
class st_model:
    def __init__(self):
        self.model = SentenceTransformer(path.abspath("../quantized_transformer_models/ST_Model"))
    
    def encode(self, text):
        return self.model.encode(text)

    def cos_sim(self, s1, s2):
        return util.pytorch_cos_sim(s1,s2).item()

    def calc_sim(self, string1, string2):
        str1_encoded = self.encode(string1)
        str2_encoded = self.encode(string2)
        return self.cos_sim(str1_encoded,str2_encoded)



ST_model = st_model()
