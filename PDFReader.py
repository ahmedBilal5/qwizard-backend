from pdfquery import PDFQuery
from io import BytesIO 

# def extractTextPDF(file):
#     text = ""
#     reader = PdfReader(BytesIO(file))
#     total = len(reader.pages)
#     for i in range(total):
#         page = reader.pages[i]
#         text += page.extract_text()
#     return text

def extractTextPDF(file):
    pdf = PDFQuery(BytesIO(file))
    pdf.load()
    text_elements = pdf.pq('LTTextLineHorizontal')
    text = [t.text for t in text_elements]
    return "".join(text)