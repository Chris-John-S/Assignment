import os
import sys


path = sys.argv[0]
dir = os.path.dirname(os.path.abspath(path))
os.chdir(dir)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        if sys.argv[1] == "text-extraction":
            img_path = sys.argv[2]
            if os.path.exists(img_path):
                from text_extraction.extraction import image_to_string
                txt = image_to_string(img_path)
                print(txt)
            else:
                print("Invalid input")
        elif sys.argv[1] == "text-analysis":
            img_path = sys.argv[2]
            if os.path.exists(img_path):
                from text_analysis.analysis import analyze
                analyze(img_path)
            else:
                print("Invalid input")
        elif sys.argv[1] == "image-classification":
            img_path = sys.argv[2]
            if os.path.exists(img_path):
                from image_classification.classification import classify
                classify(img_path)
            else:
                print("Invalid input")
        else:
            print("Invalid input")
    elif len(sys.argv) == 2:
        if sys.argv[1] == "joke-generation":
            from LLM.generation import generate
            generate()
        else:
            print("Invalid input")
    else:
        print("Invalid input")
