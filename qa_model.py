from transformers import AutoModelWithLMHead, AutoTokenizer


class qa_model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("D:/Work/Semester_7/FYP/FYP_Core_logic/Transformer_Models/QA_Model")
        self.model = AutoModelWithLMHead.from_pretrained("D:/Work/Semester_7/FYP/FYP_Core_logic/Transformer_Models/QA_Model")


    def get_answer(self, question, context):
        input_text = "context: %s <question for context: %s </s>" % (context,question)
        features = self.tokenizer([input_text], return_tensors='pt')
        out = self.model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'])
        return self.tokenizer.decode(out[0])



QA_model = qa_model()
