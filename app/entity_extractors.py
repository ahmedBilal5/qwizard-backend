from nltk import word_tokenize, pos_tag
from random import shuffle


def non_noun_entity_extractor(keywords, num):
  entities = []
  for keyword in keywords:
    tokenized_keyword = word_tokenize(keyword)
    tagged_words = pos_tag(tokenized_keyword)
    for word,tag in tagged_words:
      #if tag == 'NNP' or tag == 'NNPS' or tag == 'NN' or tag == 'RB' or tag == 'JJ' or tag == 'VB' or tag == 'CD':
      if tag == 'RB' or tag == 'RBR'or tag == 'JJ' or tag == 'VB' or tag == 'VBG' or tag == 'VBD' or tag == 'VBN' or tag == 'JJR' or tag == 'VBP' or tag == 'CD':
        if tag!= 'CC':
          entities.append(word)
  
  entities = list(set(entities))
  shuffle(entities)

  return entities[:num]


def spacy_entity_extractor(full_context, model, num):
  
  doc = model(full_context)
  list_of_ents = []
  for word in doc.ents:
    #print(word.text,word.label_)
    list_of_ents.append(word.text)

  list_of_ents = list(set(list_of_ents))
  shuffle(list_of_ents)
  if(len(list_of_ents) > num):
    return list_of_ents[:num]
  return list_of_ents



  
  