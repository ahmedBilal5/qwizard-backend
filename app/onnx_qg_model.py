from transformers import AutoTokenizer
from fastT5 import get_onnx_model, get_onnx_runtime_sessions, OnnxT5
from os import path
class onnx_qg_model:
    def __init__(self):
        #abs path: "D:/Work/Semester_7/FYP_Work/FYP/FYP_Core_logic/quantized_transformer_models/onnx_quantized_QG_model/t5-base-finetuned-question-generation-ap-encoder-quantized.onnx"
        self.path = path.abspath("../quantized_transformer_models/onnx_quantized_QG_model")
        self.encoder_path = path.join(self.path, "t5-base-finetuned-question-generation-ap-encoder-quantized.onnx")
        self.decoder_path = path.join(self.path, "t5-base-finetuned-question-generation-ap-decoder-quantized.onnx")
        self.init_decoder_path = path.join(self.path, "t5-base-finetuned-question-generation-ap-init-decoder-quantized.onnx")
        self.model_paths = self.encoder_path, self.decoder_path, self.init_decoder_path
        self.model_sessions = get_onnx_runtime_sessions(self.model_paths)
        self.model = OnnxT5(self.path, self.model_sessions)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)

    def get_question(self, answer, context, max_length=64):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.tokenizer([input_text], return_tensors='pt')

        output = self.model.generate(input_ids=features['input_ids'], 
                    attention_mask=features['attention_mask'],
                    max_length=max_length)
        return self.tokenizer.decode(output[0])


Onnx_qg_model = onnx_qg_model()
# for i in range(10):
#     print(Onnx_qg_model.get_question('Ahmed Bilal', 'Ahmed Bilal is trying his best'))  