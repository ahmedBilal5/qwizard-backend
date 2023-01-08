from nltkmodules import *
from onnx_qa_model import Onnx_qa_model
from onnx_qg_model import Onnx_qg_model
from st_model import ST_model
from eval_model import Eval_model
from entity_extractors import *
from distractor_generators import *
from qa_filter import *
from context_shortener import add_corresponding_context
import glob
from spacy_models import web_spacy_model, sci_spacy_model
from sense2vec_model import s2v
from gensim_model import model_gloVe
#from concurrent.futures import ProcessPoolExecutor


print('**************  QG Model Loaded ***********************')
print('\n\n**************  QA Model Loaded ***********************\n\n')
print('\n\n**************  ST Model Loaded ***********************\n\n')
print('\n\n**************  Eval Model Loaded ***********************\n\n')
print('\n\n*************** Gensim GloVe model loaded ****************\n\n')
print('\n\n*************** Sense2Vec model loaded ****************\n\n')
print('\n\n*************** Spacy models loaded ****************\n\n')



def extract_question(question):
  question= question[16:]
  return question[0:len(question)-4]

def find_entities_with_distractors(selected_entities,m1,m2, ST_model):
  acceptable_entities = []
  for entity_dic in selected_entities:
    
    try:
        entity_context_and_distractors = generate_distractors_gensim(entity_dic['entity'], m1, ST_model)
        entity_context_and_distractors['context'] = entity_dic['context']
        acceptable_entities.append(entity_context_and_distractors)
    except:  
      try:
        entity_context_and_distractors = generate_distractors_s2v(entity_dic['entity'], m2, ST_model)
        print('herer')
        entity_context_and_distractors['context'] = entity_dic['context']
        acceptable_entities.append(entity_context_and_distractors)
      except:
        print(entity_dic['entity'], ' - did not work')

  return acceptable_entities


def get_mcq_qa(acceptable_entities, eval_criteria):
  mcq_list = []
  for entity in acceptable_entities:
    question = Onnx_qg_model.get_question(entity['Answer'], entity['context'])
    print('Question: ', question)
    print('Answer: ', entity['Answer'])
    eval_score = Eval_model.get_evaluation(entity['Answer'], question)
    print('eval score: ', eval_score[1])
    if(eval_score[1] > eval_criteria):
      mcq_list.append((question,{'Answer': entity['Answer'], 'Distractors': entity['Distractors']} , eval_score[1]))
      print('*** appended to qa_list ***')
    print('\n\n')
  return mcq_list


def get_open_ended_qa(selected_entities_with_context):
  qa_list = []
  for entity_dic in selected_entities_with_context: 
    question = Onnx_qg_model.get_question(entity_dic['entity'], entity_dic['context'])
    answer = Onnx_qa_model.get_answer(question,entity_dic['context'])
    eval_score = Eval_model.get_evaluation(answer, question)
    print('context: ', entity_dic['context'], '\n\n')
    print('Question: ', question)
    print('Answer: ', answer)
    print('Eval score', eval_score, '\n\n')
    
    qa_list.append((question, answer, eval_score))
  return qa_list


def current_filenum(filename, q_type):
    write_directory = './generated_quizzes/*.txt'
    filenames = [file[20:] for file in glob.glob(write_directory)]

    print(filenames)
    match_str = filename.split('.')[0] + '_' + q_type
    matching_files = []
    for filen in filenames:
        if(filen.find(match_str) != -1):
            matching_files.append(filen)

    if(len(matching_files)!=0):
        file_nums = []
        for filen in matching_files:
            file_num = int(filen .split('.')[0][-1])
            file_nums.append(file_num)   

        current_file_num = max(file_nums)+1 
        return current_file_num
    else:
        return 0


def generate_mcqs(passage):
  entities_web = spacy_entity_extractor(passage,web_spacy_model, 20)
  entities_sci = spacy_entity_extractor(passage,sci_spacy_model, 40 - len(entities_web))
  entities_all = entities_web + entities_sci
  entities_with_their_context = add_corresponding_context(entities_all, passage)
  entities_and_distractors = find_entities_with_distractors(entities_with_their_context, model_gloVe, s2v, ST_model)
  print('Distractors generated')         
  mcq_qa = get_mcq_qa(entities_and_distractors, 0.0)
  #print(mcq_qa)
  print('MCQ QA generated')
  best_mcq_qa = best_n_questions(25, mcq_qa)
  #uncomment next line for uniqueness
  #unique_best_mcq_qa = unique_questions(best_mcq_qa, 'mcq', ST_model)
  unique_best_mcq_qa = best_mcq_qa
  mcq_list = []
  for i in range(len(unique_best_mcq_qa)):
    mcq_dict = {}
    unique_best_mcq_qa[i][1]['Distractors'].append(unique_best_mcq_qa[i][1]['Answer'])
    shuffle(unique_best_mcq_qa[i][1]['Distractors'])
    mcq_dict['Question'] = unique_best_mcq_qa[i][0][16:len(unique_best_mcq_qa[i][0])-4]
    mcq_dict['Options'] =  unique_best_mcq_qa[i][1]['Distractors']
    mcq_dict['Answer'] = unique_best_mcq_qa[i][1]['Answer']
    mcq_list.append(mcq_dict) 

  return mcq_list


def generate_open_ended_qa(passage):
  non_noun_entities = non_noun_entity_extractor(passage, 20)
  entities_sci = spacy_entity_extractor(passage,sci_spacy_model, 30 - len(non_noun_entities))
  entities_all = non_noun_entities + entities_sci

  entities_with_their_context = add_corresponding_context(entities_all, passage)
  open_ended_qa = get_open_ended_qa(entities_with_their_context)
  best_open_ended_qa = best_n_questions(25, open_ended_qa)
  #uncomment next line for uniqueness
  #unique_best_open_ended_qa = unique_questions(best_open_ended_qa, 'open_ended', ST_model)
  unique_best_open_ended_qa = best_open_ended_qa
  return [{'Question': qa[0][16:len(qa[0])-4], 'Answer': qa[1][6:len(qa[1])-4]} for qa in unique_best_open_ended_qa]


  

    

if __name__ == '__main__':

    #executor = ProcessPoolExecutor(max_workers = 2)
    continue_flag = 'c'
    while(continue_flag != 'q'):

        print('\n\n************* Enter filename containing text ***************************************\n')
        filename = input()
        full_path = 'D:/Work/Semester_7/FYP_Work/FYP/FYP_Core_logic/sample_text_passages/' + filename
        f = open(full_path, "r")
        passage = f.read()
        print('\n\n************* Enter type of Qs (MCQ or OPEN_ENDED) ***************************************\n')
        q_type = input()
        q_type = q_type.lower()
        write_directory = './generated_quizzes/'
        
        

        if (q_type == 'mcq'):
            
            file_num = current_filenum(filename, q_type)

            # entities_web = spacy_entity_extractor(passage,web_spacy_model, 20)
            # entities_sci = spacy_entity_extractor(passage,sci_spacy_model, 40 - len(entities_web))
            # entities_all = entities_web + entities_sci

            # entities_with_their_context = add_corresponding_context(entities_all, passage)


            # #now generating distractors
            # entities_and_distractors = find_entities_with_distractors(entities_with_their_context, model_gloVe, s2v, ST_model)
            # print('Distractors generated')
            
            # mcq_qa = get_mcq_qa(entities_and_distractors, -1.0)
            # #print(mcq_qa)
            # print('MCQ QA generated')

            # best_mcq_qa = best_n_questions(25, mcq_qa)

            # unique_best_mcq_qa = unique_questions(best_mcq_qa, q_type, ST_model)

            # for q in unique_best_mcq_qa:
            #     print('Question: ', q[0])
            #     print('Options: ', q[1]['Distractors'])
            #     print('Answer: ', q[1]['Answer'], '\n\n')


            # write_filename = write_directory + filename.split('.')[0] + '_' + q_type + str(file_num) + '.txt'
            # write_file = open(write_filename, "w") 

            # for q in unique_best_mcq_qa:
            #     write_file.write('Question: ' + q[0]+ '\n')
            #     distr = ", ".join(str(d) for d in q[1]['Distractors'])
            #     write_file.write('Options: ' + distr + '\n')
            #     write_file.write('Answer: ' + str(q[1]['Answer'])+ '\n\n')

            # write_file.close()
            # print('File created and questions written to file')
            mcqs = generate_mcqs(passage)
            print(mcqs)


        elif (q_type =='open_ended'):
            
            # file_num = current_filenum(filename, q_type)

            # non_noun_entities = non_noun_entity_extractor(passage, 20)
            # entities_sci = spacy_entity_extractor(passage,sci_spacy_model, 30 - len(non_noun_entities))
            # entities_all = non_noun_entities + entities_sci

            # entities_with_their_context = add_corresponding_context(entities_all, passage)
            # open_ended_qa = get_open_ended_qa(entities_with_their_context)
            # best_open_ended_qa = best_n_questions(25, open_ended_qa)
            # unique_best_open_ended_qa = unique_questions(best_open_ended_qa, q_type, ST_model)
            

            # for q in unique_best_open_ended_qa:
            #     print('Question: ', q[0])
            #     print('Answer: ', q[1], '\n\n')

            # write_filename = write_directory + filename.split('.')[0] + '_' + q_type + str(file_num) + '.txt'
            # write_file = open(write_filename, "w") 

            
            # for q in unique_best_open_ended_qa:
            #     write_file.write('Question: '+q[0])
            #     write_file.write('Answer: '+ q[1]+ '\n\n')

            # write_file.close()
            # print('File created and questions written to file')

            open_ended_qa = generate_open_ended_qa(passage)
            for qa in open_ended_qa:
              print(qa)

        elif q_type=='both':
            #executor = ProcessPoolExecutor(max_workers = 2)
            #open_ended_qa_process = executor.submit(generate_open_ended_qa,passage)
            mcq_qa_process = executor.submit(generate_mcqs,passage)
            full_quiz = {}
            full_quiz['MCQs'] = mcq_qa_process.result()
            full_quiz['Open ended questions'] = open_ended_qa_process.result()
            print('\n\nFinal full quiz\n\n', full_quiz) 

        else:
          continue

        print('\n\n************* Enter q to quit. Enter anything else to continue ***************************************\n')
        continue_flag = input()
        continue_flag = continue_flag.lower()

    f.close()