from huggingface_hub.inference_api import InferenceApi
inference = InferenceApi(repo_id="bert-base-uncased", token='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')
# for i in range(5):
#     print(inference(inputs="The goal of life is [MASK]."))

model = InferenceApi(repo_id='sentence-transformers/bert-base-nli-mean-tokens', token='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')
print(model.extract_keywords('blue', keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=3))