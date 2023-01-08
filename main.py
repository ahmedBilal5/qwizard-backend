from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
#from onnx_qg_model import Onnx_qg_model
#import sys
#from FYP_Core_logic.Question_Answer_generation.qa_generation import generate_mcqs
#sys.path.insert(0, 'D:\Work\Semester_7\FYP\FYP_Core_logic\Question_Answer_generation')
from qa_generation import *
import uvicorn
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://qwizard-next.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#print("all_done")


# @app.get('/')
# def home():
#     return {"Data": "Hello World"}


# @app.get('/about')
# def home():
#     return {"Data": "About"}


# inventory = {
#     1: {
#         "name": 'milk',
#         "price": 3.99,
#         "brand": "regular"
#     },
#     2: {
#         "name": 'cheese',
#         "price": 4.99,
#         "brand": "assortment"
#     }
# }

# @app.get("/get-item/{item_id}")
# def get_item(item_id: int):
#     return inventory[item_id]



class passage(BaseModel):
    content: str

class ans_and_context(BaseModel):
    answer: str
    context: str
    

class qa_pair(BaseModel):
    question: str
    answer: str

class open_ended_qa(BaseModel):
    qa: List[qa_pair]


class mcq(BaseModel):
    question: str
    options: str
    answer: str

class mcq_list(BaseModel):
    qa: List[mcq]


#making a post request to get a passage and then responding with the questions.
@app.post("/generate_mcqs")
def gen_mcq(Passage: passage):
    print(Passage.content)
    mcqs = generate_mcqs(Passage.content)
    return mcqs

@app.post("/generate_qa")
def gen_qa(Passage: passage):
    print(passage)
    qa = generate_open_ended_qa(Passage.content)
    return qa


@app.post("/generate_question")
def gen_q(Ans_and_context: ans_and_context):
    q = Onnx_qg_model.get_question(Ans_and_context.answer, Ans_and_context.context)
    return q

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)