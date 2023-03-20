from docx import Document
from io import BytesIO 

def getText(file):
    doc = Document(BytesIO(file))
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

