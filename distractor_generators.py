from typing import List, Tuple
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk import pos_tag


def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.9) -> List[Tuple[str, float]]:
    
    """ Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.


    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances

    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]



def get_answer_and_distractor_embeddings(answer,candidate_distractors,STmodel):
  answer_embedding = STmodel.encode([answer])
  distractor_embeddings = STmodel.encode(candidate_distractors)
  return answer_embedding,distractor_embeddings

#complete distractor generation function
def generate_distractors_s2v(originalword, s2v, ST_model):
  word = originalword.lower()
  word = word.replace(" ", "_")

  print ("word ",word)
  sense = s2v.get_best_sense(word)

  print ("Best sense ",sense)
  most_similar = s2v.most_similar(sense, n=20)
  print(most_similar[:5])
  #print (most_similar)

  distractors = []

  for each_word in most_similar:
    append_word = each_word[0].split("|")[0].replace("_", " ")
    if append_word not in distractors and append_word != originalword:
        distractors.append(append_word)

  distractors.insert(0,originalword)
  answer_embedd, distractor_embedds = get_answer_and_distractor_embeddings(originalword,distractors, ST_model)

  final_distractors = mmr(answer_embedd,distractor_embedds,distractors,5)
  filtered_distractors = []
  for dist in final_distractors:
    filtered_distractors.append (dist[0])

  Answer = filtered_distractors[0]
  Filtered_Distractors =  filtered_distractors[1:]

  options_dic = {}
  #print (Answer)
  options_dic['Answer'] = Answer
  print ("------------------->")
  distractor_lst  = []
  for k in Filtered_Distractors:
    distractor_lst.append(k)
  options_dic['Distractors'] = distractor_lst
  return options_dic


def generate_distractors_gensim(originalword, model_glove, ST_model):
  word = originalword.lower()
  word = word.replace(" ", "_")

  print ("using gensim ")

  most_similar = model_glove.most_similar(word, topn=40)

  #print (most_similar)

  all_distractors = []

  for each_word in most_similar:
    all_distractors.append(each_word[0])
  
  #print(all_distractors)

  original_word_pos_tag = pos_tag([originalword])[0][1]
  all_distractors_pos_tag = pos_tag(all_distractors)


  distractors= []
  for each_distractor in all_distractors_pos_tag:
    if(each_distractor[1]==original_word_pos_tag):
      distractors.append(each_distractor[0])
    
  #print(distractors)
  distractors.insert(0,originalword)
  answer_embedd, distractor_embedds = get_answer_and_distractor_embeddings(originalword,distractors, ST_model)

  final_distractors = mmr(answer_embedd,distractor_embedds,distractors,5)
  filtered_distractors = []
  for dist in final_distractors:
    filtered_distractors.append (dist[0])

  Answer = filtered_distractors[0]
  Filtered_Distractors =  filtered_distractors[1:]

  options_dic = {}
  #print (Answer)
  options_dic['Answer'] = Answer
  print ("------------------->")
  distractor_lst  = []
  for k in Filtered_Distractors:
    distractor_lst.append(k)
  options_dic['Distractors'] = distractor_lst
  print(distractor_lst)
  return options_dic



# def find_entities_with_distractors(selected_entities):
#   acceptable_entities = []
#   for entity_dic in selected_entities:
#     try:
#         entity_context_and_distractors = generate_distractors_gensim(entity_dic['entity'])
#         entity_context_and_distractors['context'] = entity_dic['context']
#         acceptable_entities.append(entity_context_and_distractors)
#     except:
#       try:
#         entity_context_and_distractors = generate_distractors_s2v(entity_dic['entity'])
#         entity_context_and_distractors['context'] = entity_dic['context']
#         acceptable_entities.append(entity_context_and_distractors)
#       except:
#         print(entity_dic['entity'], ' - did not work')
#   return acceptable_entities