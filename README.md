# qwizard-backend

![alt text](https://qwizard-next.vercel.app/logo.svg)

## This repository holds the code for the backend of QWizard, in python.
## This repository includes:

 * The code for Question and Answer generation from text.
 * The code for the FAST API server.

Inside the _app_ folder:

* The _main.py_ file contains the FAST-API server code. 
* The _qa_generation.py_ contains the code for Question and Answer generation.
* _context_shortener.py_, _distractor_generation.py_ and _entity_extractors.py_ contain the various different helping functions and text processing functions for Question and Answer Generation
* All the _.py_ files whose names end with _model_, load the models from the disk:
  * _qa_model.py_: Transformer model for Question Answering.
  * _qg_model.py_: Transformer model for Question Generation.
  * _eval_model.py_: Transformer model for Question Answer pair evaluation.
  * _st_model.py_: Sentence Transformer model for calculating BERT embeddings for sentences.

Due to the __large size__ of the Transformer models (> 1GB), the models themselves could not be pushed to the repository.
The python package requirements are included in the __requirements.txt__ file.
