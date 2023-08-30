# Assignment

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Text Extraction](#text-extraction)
- [Text Analysis](#text-analysis)
- [Image Classification](#image-classification)
- [Joke Generation](#joke-generation)
- [Collaboration Simulation](collaboration/collaboration.md)

## Introduction
This is an assignment - Text extraction, text analysis, image classification, joke generation, collaboration simulation.

## Installation
Install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```


## Text-Extraction

Given any set of scanned documents:
- Use an OCR library to extract the text from the documents. 
- Clean and preprocess the extracted text data for further analysis.

`python run.py text-extraction ./sample_image/sample.jpg`

For text extraction, used tesseract-ocr which is the state-of-the-art net.

# Text-Analysis

- Identify named entities in the text extracted above using OCR. 
- Classify sentences based on sentiment (positive, negative, neutral).

`python run.py text-analysis ./sample_image/sample.jpg`

For text analysis, processed the extracted text with spaCy for named entity recognition and analyzed sentiment for each sentence with textblob.

# Image-Classification

Given a dataset of images(keep it down to 3 max):
- Develop a basic model to classify the images into predefined categories.
- You can use any pre-trained model or open-source implementation.

`python run.py image-classification ./sample_image/pic1.jpg`
`python run.py image-classification ./sample_image`

For image classification, used pretrained MobileNetV3-small and added Linear layer for final purpose.

# Joke-Generation

Using any available LLM (like GPT-2 or similar):
- Train the model : on any dataset
- Generate a short paragraph on a given prompt

`python run.py joke-generation`

I trained joke generation model using GPT-2.

# Collaboration-Simulation

Document a hypothetical situation where an issue arose in your code, and describe how you would collaborate with a teammate to resolve it.