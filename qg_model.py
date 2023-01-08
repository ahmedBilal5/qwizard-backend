from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastT5 import get_onnx_model, get_onnx_runtime_sessions, OnnxT5

class qg_model:
    def __init__(self):
        self.encoder_path = "D:/Work/Semester_7/FYP Work/FYP/FYP_Core_logic/quantized_transformer_models/onnx_quantized_QG_model"
        self.model = AutoModelForSeq2SeqLM.from_pretrained("D:/Work/Semester_7/FYP/FYP_Core_logic/Transformer_Models/QG_Model")

    def get_question(self, answer, context, max_length=64):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.tokenizer([input_text], return_tensors='pt')

        output = self.model.generate(input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=max_length)
        return self.tokenizer.decode(output[0])



QG_model = qg_model()

