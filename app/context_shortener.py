from nltk import sent_tokenize

second_article = """ Marie Curie was one of the most accomplished scientists in history. Together with her husband, Pierre, she discovered radium, an element widely used for treating cancer, and studied uranium and other radioactive substances. Pierre and Marie’s amicable collaboration later helped to unlock the secrets of the atom.

Marie was born in 1867 in Warsaw, Poland, where her father was a professor of physics. At an early age, she displayed a brilliant mind and a blithe personality. Her great exuberance for learning prompted her to continue with her studies after high school. She became disgruntled, however, when she learned that the university in Warsaw was closed to women. Determined to receive a higher education, she defiantly left Poland and in 1891 entered the Sorbonne, a French university, where she earned her master’s degree and doctorate in physics.

Marie was fortunate to have studied at the Sorbonne with some of the greatest scientists of her day, one of whom was Pierre Curie. Marie and Pierre were married in 1895 and spent many productive years working together in the physics laboratory. A short time after they discovered radium, Pierre was killed by a horse-drawn wagon in 1906. Marie was stunned by this horrible misfortune and endured heartbreaking anguish. Despondently she recalled their close relationship and the joy that they had shared in scientific research. The fact that she had two young daughters to raise by herself greatly increased her distress.

Curie’s feeling of desolation finally began to fade when she was asked to succeed her husband as a physics professor at the Sorbonne. She was the first woman to be given a professorship at the world-famous university. In 1911 she received the Nobel Prize in chemistry for isolating radium. Although Marie Curie eventually suffered a fatal illness from her long exposure to radium, she never became disillusioned about her work. Regardless of the consequences, she had dedicated herself to science and to revealing the mysteries of the physical world. """


def get_index_of_sentence_token(sentence_tokens, keyword):
  for i in range(0,len(sentence_tokens)):
    found = -1
    found = sentence_tokens[i].find(keyword)
    if(found != -1):
      return i
  return -1


def shortened_context(full_context, keyword, num_sents, after):
  sentence_tokens = sent_tokenize(full_context)
  index_of_relevant_token = get_index_of_sentence_token(sentence_tokens, keyword)
  if(index_of_relevant_token != -1):
    selected_sentences = []
    if(index_of_relevant_token-num_sents >= 0 and index_of_relevant_token+num_sents < len(sentence_tokens)):  
      
      for i in range(num_sents,-1,-1):
        selected_sentences.append(sentence_tokens[index_of_relevant_token-i])
      if(after):  
        for i in range(1,num_sents+1):
          selected_sentences.append(sentence_tokens[index_of_relevant_token+i])

    # elif(index_of_relevant_token-num_sents < 0 and index_of_relevant_token+num_sents >= len(sentence_tokens)):
    #   raise ValueError('Context is already shorter than', 2*num_sents+2, ' sentences.')
    
    elif(index_of_relevant_token-num_sents < 0):
      for i in range(0,index_of_relevant_token+1):
         selected_sentences.append(sentence_tokens[i])
      if(after):   
        for i in range(1,num_sents+1):
          selected_sentences.append(sentence_tokens[index_of_relevant_token+i])
      
    elif(index_of_relevant_token+num_sents >= len(sentence_tokens)):
      for i in range(num_sents,-1,-1):
        selected_sentences.append(sentence_tokens[index_of_relevant_token-i])
      if(after):
        for i in range(index_of_relevant_token+1,len(sentence_tokens)):
          selected_sentences.append(sentence_tokens[i])
    
  shortened_context = ""
  for sent in selected_sentences:
    shortened_context += sent
    shortened_context += ' '
    
      
  return shortened_context[:len(shortened_context)-1]


def add_corresponding_context(selected_entities, full_context):
  selected_entities_with_corresponding_context = []
  for entity in selected_entities:
    entity_dic = {}
    entity_dic['entity'] = entity
    entity_dic['context'] = shortened_context(full_context, entity, 4, False)
    selected_entities_with_corresponding_context.append(entity_dic)
  return selected_entities_with_corresponding_context
  


