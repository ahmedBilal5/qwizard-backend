
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
import torch.nn.functional as F
import nltk

path = __file__
file_name=path.split("\\")
dir=""
for i in range(len(file_name)-1):
    dir=dir+file_name[i]+'\\' 


# nltk.download("wordnet", "C:\\nltk_data\\")
# from nltk.corpus import wordnet

# def methodC(sentence1, sentence2):
#     # Tokenize the sentences
#     tokens1 = nltk.word_tokenize(sentence1)
#     tokens2 = nltk.word_tokenize(sentence2)

#     # Get the synset for each token
#     synsets1 = [wordnet.synsets(token) for token in tokens1]
#     synsets2 = [wordnet.synsets(token) for token in tokens2]

#     # Flatten the list of synsets
#     synsets1 = [synset for sublist in synsets1 for synset in sublist]
#     synsets2 = [synset for sublist in synsets2 for synset in sublist]

#     # Calculate the semantic similarity score
#     score = 0.0
#     for synset1 in synsets1:
#         for synset2 in synsets2:
#             similarity = synset1.wup_similarity(synset2)
#             if similarity is not None:
#                 score += similarity
#     # return score
#     # Return the average score
#     return score / (len(synsets1) * len(synsets2))



# #Method 4. This code uses the TensorFlow library to create a simple neural network that takes as
# # input the average of the word embeddings for each sentence and predicts the semantic similarity
# # between the two sentences.
# # The word embeddings are a set of numerical vectors that represent the meanings of words in a
# # high-dimensional space.
# # You can use a pre-trained word embedding model, such as GloVe or Word2Vec, to generate 
# # the word embeddings for your data.

# # Load the word embeddings
# # word_embeddings = np.load(dir+"word_embeddings.npy")
# # word_embeddings = np.load(dir+"glove.6B.50d.txt")

# # # Create a simple neural network
# # model = tf.keras.Sequential()
# # model.add(tf.keras.layers.Dense(64, input_shape=(300,), activation='relu'))
# # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# # # Compile the model
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # def compare_sentences(sentence1, sentence2):
# #     # Tokenize the sentences
# #     tokens1 = nltk.word_tokenize(sentence1)
# #     tokens2 = nltk.word_tokenize(sentence2)

# #     # Get the word embeddings for each token
# #     embeddings1 = [word_embeddings[token] for token in tokens1]
# #     embeddings2 = [word_embeddings[token] for token in tokens2]

# #     # Average the word embeddings to get a sentence embedding
# #     embedding1 = np.mean(embeddings1, axis=0)
# #     embedding2 = np.mean(embeddings2, axis=0)

# #     # Use the neural network to predict the similarity between the two sentences
# #     prediction = model.predict([[embedding1], [embedding2]])
# #     score = prediction[0][0]

# #     return score

# # # Test the function
# # sentence1 = "The cat sat on the mat"
# # sentence2 = "The feline was lounging on the rug"
# # score = compare_sentences(sentence1, sentence2)
# # print(f"Semantic similarity score: {score:.2f}")



# # import numpy as np
# # from scipy import spatial
# # import matplotlib.pyplot as plt
# # from sklearn.manifold import TSNE

# # fsize=0

# def loadingGloveEmbeddings(gloveFile="glove.6B.50d.txt"):
#     with open(dir+gloveFile, 'r',  encoding="utf8") as f:
#         for line in f:
#             fsize=fsize+1
#     f.close()

#     embeddings_dict = {}
#     current=0
#     with open(dir+gloveFile, 'r',  encoding="utf8") as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             vector = np.asarray(values[1:], "float32")
#             embeddings_dict[word] = vector
#             current=current+1
#             if (current%500 == 0):
#                 print(current,"\t/",fsize,end="\r")
#     f.close()
#     return embeddings_dict

# count=0
# for i in embeddings_dict.keys():
#     print(i," : ",embeddings_dict[i])
#     count=count+1
#     if count==50:
#         break

def cosine_similarity(A=[],B=[]):
    from math import sqrt
    sizeA=len(A)
    sizeB=len(B)
    if sizeA!=sizeB:
        difference=sizeA-sizeB
        if difference<0:
            difference=difference*-1
        if sizeA>sizeB:
            for i in range(0,difference):
                B.append(0)
        else:
            for i in range(0,difference):
                A.append(0)
    
    Length_A=0
    Length_B=0
    dot_product=0
    
    for i in range(0,len(A)):
        dot_product = dot_product + (A[i]*B[i])
    
    for i in range(0,len(A)):
        Length_A = Length_A + (A[i]**2)
    Length_A=sqrt(Length_A)
    
    for i in range(0,len(B)):
        Length_B = Length_B + (B[i]**2)
    Length_B=sqrt(Length_B)
    
    try:
        cosine_sim = (dot_product / (Length_A*Length_B))
        return cosine_sim
    except:
        return 0


# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["prince"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["queen"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["uncle"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["ii"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["grandson"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["man"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["she"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["women"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["is"]))
# print("Sim: ", cosine_similarity(embeddings_dict["king"],embeddings_dict["king"]))

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(64, input_shape=(300,), activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# import nltk
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# def compare_sentences(sentence1, sentence2):
#     # Tokenize the sentences
#     sentence1=sentence1.lower()
#     sentence2=sentence2.lower()
#     tokens1 = nltk.word_tokenize(sentence1)
#     tokens2 = nltk.word_tokenize(sentence2)

#     # Get the word embeddings for each token
#     embeddings1 = [embeddings_dict[token] for token in tokens1]
#     embeddings2 = [embeddings_dict[token] for token in tokens2]

#     # Average the word embeddings to get a sentence embedding
#     embedding1 = np.mean(embeddings1, axis=0)
#     embedding2 = np.mean(embeddings2, axis=0)

#     # Use the neural network to predict the similarity between the two sentences
#     prediction = model.predict([[embedding1], [embedding2]])
#     score = prediction[0][0]

#     return score

# # Test the function
# sentence1 = "The cat sat on the mat"
# sentence2 = "The feline was lounging on the rug"
# score = compare_sentences(sentence1, sentence2)
# print(f"Semantic similarity score: {score:.2f}")

#Method 5
# sentences = [
#     "John likes the subject of Natural Language Processing.",
#     # "John likes the subject of Software for Mobile Development.",
#     # "John dislikes the subject of Natural Language Processing.",
#     # "John hates the subject of Natural Language Processing."
#     # "John loves Natural Language Processing.",
#     "John dont like Natural Language Processing. He prefers civil engineering more."
# ]

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bert-base-nli-mean-tokens')
# sentence_embeddings = model.encode(sentences)
# print(sentence_embeddings.shape)
# print()
# print(sentence_embeddings)
# print()

# from sklearn.metrics.pairwise import cosine_similarity
# x=cosine_similarity(
#     [sentence_embeddings[0]],
#     sentence_embeddings[1:]
# )

# print(x)



#Method 5, BERT Embeddings, Cosine Similarity, Spacy Transformers for NER and keyBERT for Key-Word Extraction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')
def sementicSimilarity(correctAnswer, studentAnswer):
    correntAnswerEmbeddings = model.encode(correctAnswer)
    studentAnswerEmbeddings = model.encode(studentAnswer)  
    score=cosine_similarity(
        [correntAnswerEmbeddings],
        [studentAnswerEmbeddings]
    )
    return score[0][0]

from spacy_models import web_spacy_model as nlp
def entityScore(correctAnswer, studentAnswer):
    correctAnswer=correctAnswer.replace("'s","")
    studentAnswer=studentAnswer.replace("'s","")
    correntSentence = nlp(correctAnswer)
    studentSentence = nlp(studentAnswer)  
    score=0

    cEntities=(correntSentence.ents)
    sEntities=(studentSentence.ents)
    correntEntities=[]
    studentEntities=[]
    for i in cEntities:
        correntEntities.append(str(i))
    for i in sEntities:
        studentEntities.append(str(i))

    # print("Valid Entities: ", correntEntities)
    # print("Provided Entires: ", studentEntities)
    # print()
    for i in correntEntities:
        if i in studentEntities:
            score=score+1

    return (score/len(correntEntities))

from keybert import KeyBERT
kw_model = KeyBERT(model='all-mpnet-base-v2')
def keyWordScore(correctAnswer, studentAnswer):
    keywords = kw_model.extract_keywords(correctAnswer, keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=7)
    correctKeyWords= list(dict(keywords).keys())
    keywords = kw_model.extract_keywords(studentAnswer, keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=7)
    studentKeyWords= list(dict(keywords).keys())
    score=0

    # print("Valid keyWords: ", correctKeyWords)
    # print("Provided keyWords: ", studentKeyWords)
    # print()

    for i in correctKeyWords:
        if i in studentKeyWords:
            score=score+1
    return (score/len(correctKeyWords))


from datasets import load_metric
import evaluate
def rougeScore(correctAnswer, studentAnswer):
    rouge = evaluate.load("rouge")
    predictions = [studentAnswer]
    references = [correctAnswer]
    x=rouge.compute(predictions=predictions, references=references)
    return x['rougeLsum']

def getNumberOfKeywords(sentence):
    keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1,1), stop_words='english', highlight=False, top_n=7)
    KeyWordsList = list(dict(keywords).keys())
    return len(KeyWordsList)

def getNumberOfEntities(Answer):
    Answer=Answer.replace("'s","")
    Sentence = nlp(Answer)
    cEntities=(Sentence.ents)
    Entities=[]
    for i in cEntities:
        Entities.append(str(i))
    return len(Entities)


def answerEvaluation(correctAnswer, studentAnswer,totalmarks=1):
    
    numberOfCorrectKeywords=getNumberOfKeywords(correctAnswer)
    numberOfCorrectEntities=getNumberOfEntities(correctAnswer)
   
    rougeWeightage=0.1
        
    NER_Weightage=0.05
    if (numberOfCorrectEntities==0):
        NER_Weightage=0.0

    keyWordWeightage=0.05
    if (numberOfCorrectKeywords==0):
        keyWordWeightage=0.0


    sementicWeightage=(1.0-(rougeWeightage+NER_Weightage+keyWordWeightage))
    
    sementicPoints=sementicSimilarity(correctAnswer,studentAnswer)
    
    # print("Sementic Similarity: ", sementicPoints)
    # print()
    entityPoints=0
    keywordPoints=0
    if (numberOfCorrectEntities>0):
        entityPoints=entityScore(correctAnswer,studentAnswer)
    if (numberOfCorrectKeywords>0):
        keywordPoints=keyWordScore(correctAnswer,studentAnswer)  
         
    roughPoints=rougeScore(correctAnswer,studentAnswer)

    score=0
    score=score+(sementicWeightage*sementicPoints)
    score=score+(NER_Weightage*entityPoints)
    score=score+(keyWordWeightage*keywordPoints)
    score=score+(rougeWeightage*roughPoints)
    # score=score*totalmarks

    # print(sementicWeightage, "\tx\t", sementicPoints, "\t: ", (sementicWeightage*sementicPoints))
    # print(NER_Weightage, "\tx\t", entityPoints, "\t\t: ", (NER_Weightage*entityPoints))
    # print(keyWordWeightage, "\tx\t", keywordPoints, "\t\t: ", (keyWordWeightage*keywordPoints))
    # print(rougeWeightage, "\tx\t", roughPoints, "\t\t: ", (rougeWeightage*roughPoints))
    # print("Total: \t", score)
    # print()

    correctBoundary=0.7
    okBoundary=0.35
    badBoundary=0.25
    
    if (score>=correctBoundary):
        return (1.0*totalmarks)
    elif ((score>=okBoundary) and (score<correctBoundary)):
        return ((score/correctBoundary)*totalmarks)
    elif ((score>=badBoundary) and (score<okBoundary)):
        return (score*totalmarks)
    elif (score<badBoundary):
        return 0

    
    return (score*totalmarks)




#print(answerEvaluation("It is hot and arid", "It is not hot and arid at all"))

# print(answerEvaluation("Samsung has unveiled what is undoubtedly its largest Android smartphone to date, the Galaxy Mega.","Samsung unveils its largest Android smartphone Galaxy Mega",2))
# print(answerEvaluation("Apple has paid around GBP 3 million to buy 8.2 million shares in UK-based digital radio maker Imagination Technologies.","Apple buys into digital radio maker Imagination Technologies",2))
# print(answerEvaluation("US President Barack Obama extended sanctions against Belarus for another consecutive year, the White House said on its website on Wednesday.","Obama extends sanctions against Belarus",2))
# print(answerEvaluation("Nigel Hasselbaink is back in contention for Hamilton`s Scottish Cup clash against Alloa after recovering from a knee injury.","Hasselbaink could be back",2))
# print(answerEvaluation("Washington, Feb 5 Hitting the road once again in his trademark style, the US President Barack Obama has said it is ``time to do something'' to prevent the kind of deadly shooting incidents that has been happening in the country killing innocent people, including children.","It is time to do something",2))


# sen1="Google provides very good search results"
# sen2="It is a optimised search engine"
# print(sen1,"\n",sen2,"\n",answerEvaluation(sen1,sen2),"\n")
# print("--------------------------------------\n\n")



# from datasets import load_dataset
# dataset = load_dataset("embedding-data/sentence-compression")

# from csv import writer
# import pickle

# import time
# from os.path import exists

# file_exists = exists(dir+"results.csv")
# if (file_exists==False):
#     f = open(dir+"results.csv", "x")
#     f.close()

# with open(dir+"results.csv", 'a', encoding="utf8") as f_object:
#     writer_object = writer(f_object)
#     st = time.time()
#     for i in range(33000,33500):
#     # for i in range(0,1):
#         # if i in done:
#         #     continue
        
#         sen1=dataset["train"][i]["set"][0]
#         sen2=dataset["train"][i]["set"][1]
#         # print("R: ", rougeScore(sen1,sen2))
#         data=[i,sen1, sen2, answerEvaluation(sen1,sen2)]
#         writer_object.writerow(data)
#         if (i%10 == 0):
#             print(i)
#             et = time.time()
#             elapsed_time = et - st
#             print('Execution time:', elapsed_time, 'seconds')
#             print()
#         # print(i)
#         # print(dataset["train"][i]["set"][0])
#         # print(dataset["train"][i]["set"][1])

#         # done.append(i);
#     et = time.time()
#     elapsed_time = et - st
#     print('Execution time:', elapsed_time, 'seconds')
#     # dbfile = open('doneFile', 'ab')
#     # pickle.dump(dbfile, done)                     
#     # dbfile.close()

#     f_object.close()


# print(answerEvaluation("Samsung has unveiled what is undoubtedly its largest Android smartphone to date, the Galaxy Mega.","Samsung unveils its largest Android smartphone Galaxy Mega",2))
# print(answerEvaluation("Apple has paid around GBP 3 million to buy 8.2 million shares in UK-based digital radio maker Imagination Technologies.","Apple buys into digital radio maker Imagination Technologies",2))
# print(answerEvaluation("US President Barack Obama extended sanctions against Belarus for another consecutive year, the White House said on its website on Wednesday.","Obama extends sanctions against Belarus",2))
# print(answerEvaluation("Nigel Hasselbaink is back in contention for Hamilton`s Scottish Cup clash against Alloa after recovering from a knee injury.","Hasselbaink could be back",2))
# print(answerEvaluation("Washington, Feb 5 Hitting the road once again in his trademark style, the US President Barack Obama has said it is ``time to do something'' to prevent the kind of deadly shooting incidents that has been happening in the country killing innocent people, including children.","It is time to do something",2))
