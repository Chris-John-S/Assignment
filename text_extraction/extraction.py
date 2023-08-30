import os
import sys
import subprocess


# Using Tesseract-OCR, perform text extraction
def image_to_string(path):
    # tesseract_path
    tesseract_cmd_small = os.path.abspath('./text_extraction/tesseract-ocr/eng.exe')

    # absolute path of image
    img_path = os.path.abspath(path)

    # path of text file to save
    txt_path = 'C:/Users/Potter/AppData/Local/Temp/' + os.path.basename(img_path).rpartition('.')[0]

    # run tesseract
    cmd_args = [tesseract_cmd_small, img_path, txt_path, "txt"]
    try:
        subprocess.call(cmd_args)
    except OSError as e:
        print("Error")

    # read textfile and return
    txt_path += ".txt"
    txt = open(txt_path).read()
    os.remove(txt_path)
    return txt


if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "sample.jpg"
    else:
        img = sys.argv[1]

    txt = image_to_string(path)
    print(txt)
