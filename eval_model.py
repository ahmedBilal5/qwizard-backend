from transformers import AutoTokenizer, AutoModelForSequenceClassification
from os import path
import torch

class eval_model:
    
    def __init__(self):
        self.path = path.abspath("../quantized_transformer_models/Eval_Model")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path)

    
    def get_evaluation(self, answer, question, max_length=64):
        inputs = self.tokenizer(text=question, text_pair=answer, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits

        #print('logits', logits[0][1].item())
        predicted_class_id = logits.argmax().item()
        #print('predicted class id', predicted_class_id)
        return (self.model.config.id2label[predicted_class_id],logits[0][1].item())


Eval_model = eval_model()


    