from transformers import AutoTokenizer
from os import path
from fastT5 import get_onnx_model, get_onnx_runtime_sessions, OnnxT5

class onnx_qa_model:
    def __init__(self):
        #abs path: "D:/Work/Semester_7/FYP_Work/FYP/FYP_Core_logic/quantized_transformer_models/onnx_quantized_qa_model/t5_abs_qa-encoder-quantized.onnx"
        self.path = path.abspath('../quantized_transformer_models/onnx_quantized_qa_model')
        self.encoder_path = path.join(self.path, "t5_abs_qa-encoder-quantized.onnx")
        self.decoder_path = path.join(self.path, "t5_abs_qa-decoder-quantized.onnx")
        self.init_decoder_path = path.join(self.path, "t5_abs_qa-init-decoder-quantized.onnx")
        self.model_paths = self.encoder_path, self.decoder_path, self.init_decoder_path
        self.model_sessions = get_onnx_runtime_sessions(self.model_paths)
        self.model = OnnxT5(self.path, self.model_sessions)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

   
    def get_answer(self, question, context):
        input_text = "context: %s <question for context: %s </s>" % (context,question)
        features = self.tokenizer([input_text], return_tensors='pt')
        out = self.model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
        return self.tokenizer.decode(out[0])


Onnx_qa_model = onnx_qa_model()
# for i in range(10):
#     print(Onnx_qa_model.get_answer('Who is trying his best?', 'Ahmed Bilal is trying his best'))  
 