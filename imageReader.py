#pip install pytesseract
#pip install PIL
#For windows, also install pytesseract.exe via https://github.com/UB-Mannheim/tesseract/wiki
#WARNING: Tesseract should be either installed in the directory which is suggested during the installation or in a new directory. The uninstaller removes the whole installation directory. If you installed Tesseract in an existing directory, that directory will be removed with all its subdirectories and files.

from PIL import Image
from pytesseract import pytesseract

#imageToText(imagePath, Optional Parameter for tesseract.exe path | defaultPath already set)


def imageToText(imagePath, path_to_tesseract=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"):
    try:
        img = Image.open(imagePath)
        try:
            pytesseract.tesseract_cmd = path_to_tesseract
            try:
                text = pytesseract.image_to_string(img)
                return text[:-1]
            except:
                print("Couldn't extract text")
                return ""
        except:
            print("Tesseract not found.")
            return ""
    except:
        try:
            path = __file__
            file_name=path.split("\\")
            dir=""
            for i in range(len(file_name)-1):
                dir=dir+file_name[i]+'\\' 

            img = Image.open(dir+imagePath)
            try:
                pytesseract.tesseract_cmd = path_to_tesseract
                try:
                    text = pytesseract.image_to_string(img)
                    return text[:-1]
                except:
                    print("Couldn't extract text")
                    return ""
            except:
                print("Tesseract not found.")
                return ""
        except:
            print("Couldn't open the image file. Check file path and extension.")
            return ""
    

# a=imageToText("test4.png")
# print(a)

