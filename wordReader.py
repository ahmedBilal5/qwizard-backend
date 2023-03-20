#pip install aspose-words

import aspose.words as aw

def docToText_Method_1(inputfile, outputFile="Output.txt"):
    if inputfile.find('.docx')== -1:
        inputfile=inputfile+".docx"
    
    if outputFile.find('.txt') == -1:
        outputFile=outputFile+'.txt'
    
    try:
        doc = aw.Document(inputfile)
        try:   
            doc.save(outputFile)
            try:
                f = open(outputFile, "r")
                data=f.read()
                f.close()

                f = open(outputFile, "w")
                data=data.replace("Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit:","")
                data=data.replace("https://products.aspose.com/words/","")
                data=data.replace("Evaluation Only. Created with Aspose.Words. Copyright 2003-2021 Aspose Pty Ltd.","")
                f.write(data)
                f.close()
            except:
                print("Couldn't clean the data")
        except:
            print("Couldn't save the text file")
    except:
        path = __file__
        file_name=path.split("\\")
        dir=""
        for i in range(len(file_name)-1):
            dir=dir+file_name[i]+'\\' 

        try:
            doc = aw.Document(dir+inputfile)
            try:   
                doc.save(dir+outputFile)
                try:
                    f = open(dir+outputFile, "r")
                    data=f.read()
                    f.close()

                    f = open(dir+outputFile, "w")
                    data=data.replace("Created with an evaluation copy of Aspose.Words. To discover the full versions of our APIs please visit:","")
                    data=data.replace("https://products.aspose.com/words/","")
                    data=data.replace("Evaluation Only. Created with Aspose.Words. Copyright 2003-2021 Aspose Pty Ltd.","")
                    f.write(data)
                    f.close()
                except:
                    print("Couldn't clean the data")
            except:
                print("Couldn't save the text file")
        except:
            print("Couldn't open the docx file")

#docToText_Method_1("test","output")















#Ignore code below. Alternative approaches test.

#  import docxpy
# file = 'file.docx'
# # extract text
# text = docxpy.process(file)
# # extract text and write images in /tmp/img_dir
# text = docxpy.process(file, "/tmp/img_dir")
# # hyperlinks
# doc = docxpy.DOCReader(file)
# doc.process()  # process file
# hyperlinks = doc.data['links']



# from docx.document import Document as _Document
# from docx.oxml.text.paragraph import CT_P
# from docx.oxml.table import CT_Tbl
# from docx.table import _Cell, Table, _Row
# from docx.text.paragraph import Paragraph
# import docx
# path = './test.docx'
# doc = docx.Document(path)

# def iter_block_items(parent):
#     if isinstance(parent, _Document):
#         parent_elm = parent.element.body
#     elif isinstance(parent, _Cell):
#         parent_elm = parent._tc
#     elif isinstance(parent, _Row):
#         parent_elm = parent._tr
#     else:
#         raise ValueError("something's not right")
#     for child in parent_elm.iterchildren():
#         if isinstance(child, CT_P):
#             yield Paragraph(child, parent)
#         elif isinstance(child, CT_Tbl):
#             yield Table(child, parent)

# for block in iter_block_items(doc):
#     # read Paragraph
#     if isinstance(block, Paragraph):
#         print(block.text)
#     # read table
#     elif isinstance(block, Table):
#         print(block.style.name)




# How about this way to list picture and text? But I don’t know how to convert wmf and emf to other formats

# from docx import Document
# from os.path import basename
# import re
# file_name = "D:/2.docx"
# doc = Document(file_name)
# a = list()
# pattern = re.compile('rId\d+')
# for graph in doc.paragraphs:
#     b = list()
#     for run in graph.runs:
#         if run.text != '':
#             b.append(run.text)
#         else:
#             # b.append(pattern.search(run.element.xml))
#             contentID = pattern.search(run.element.xml).group(0)
#             try:
#                 contentType = doc.part.related_parts[contentID].content_type
#             except KeyError as e:
#                 print(e)
#                 continue
#             if not contentType.startswith('image'):
#                 continue
#             imgName = basename(doc.part.related_parts[contentID].partname)
#             imgData = doc.part.related_parts[contentID].blob
#             b.append(imgData)
#     a.append(b)



