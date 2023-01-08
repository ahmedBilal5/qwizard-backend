from sentence_transformers import util
from copy import deepcopy


def unique_questions(qa_list,type, ST_model):
  unique_qa_list = []
  print('len of original list: ', len(qa_list))

  for qa in qa_list:
    if len(unique_qa_list) >= 1:
      print('len of unique_qa_list: ', len(unique_qa_list))
      unique = True
      for unique_qa in unique_qa_list:
        if(type=='open_ended'):
            q_sim = ST_model.calc_sim(qa[0], unique_qa[0])
            a_sim = ST_model.calc_sim(qa[1], unique_qa[1])

            if (q_sim > 0.75 and a_sim > 0):
                unique = False 


        if(type=='mcq'): 
            q_sim = ST_model.calc_sim(qa[0], unique_qa[0])

            if (q_sim > 0.75):
                unique = False


      if(unique):
        unique_qa_list.append(qa)       
    else:
      unique_qa_list.append(qa)
  
  return unique_qa_list




def best_n_questions(n,orig_qa_list):
  qa_list = deepcopy(orig_qa_list)
  best_q_list = []
  for i in range(n):
    score_list = [entry[2] for entry in qa_list]
    try:
      index_of_best_q = score_list.index(max(score_list))
      qa = qa_list.pop(index_of_best_q)
    except:
      print('No more elements. Breaking out.')
      return best_q_list
    if qa[1] != '<pad> No answer available in context</s>':
      best_q_list.append(qa)
    else:
      i-=1

  return best_q_list


