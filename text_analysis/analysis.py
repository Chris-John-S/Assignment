import sys
import spacy
from textblob import TextBlob

from text_extraction.extraction import image_to_string

def analyze(image_path):
    # Load the English language model for spaCy
    nlp = spacy.load("en_core_web_sm")

    # Extract text from image
    extracted_text = image_to_string(image_path)
    print(extracted_text, '\n')

    # Process the extracted text with spaCy for named entity recognition
    doc = nlp(extracted_text)

    # Iterate through entities and print their text and label
    for ent in doc.ents:
        print(f"Text: {ent.text}, Label: {ent.label_}")
    print('\n')

    # Get sentences.
    sentences = [s for s in extracted_text.splitlines() if s.strip() != ""]

    # Analyze sentiment for each sentence with textblob
    for sentence in sentences:
        blob = TextBlob(sentence)
        sentiment_score = blob.sentiment.polarity

        if sentiment_score > 0:
            sentiment = "positive"
        elif sentiment_score < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        print(f"Sentence: {sentence}")
        print(f"Sentiment: {sentiment}\n")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "./sample.jpg"
    else:
        img = sys.argv[1]

    analyze(path)