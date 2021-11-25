from typing import Dict
import nltk


from models import Language
from file_reader import FileReader
from analysers import (
    WordFrequencyAnalyzer,
    NeuralMethodAnalyzer,
    AlphabetMethodAnalyzer
)


def train():
    en_text = FileReader("training_samples/en/").get_text()
    es_text = FileReader("training_samples/es/").get_text()

    WordFrequencyAnalyzer.train(language=Language.ENGLISH, text=en_text)
    WordFrequencyAnalyzer.train(language=Language.SPANISH, text=es_text)

    NeuralMethodAnalyzer.train(language=Language.ENGLISH, text="en")
    NeuralMethodAnalyzer.train(language=Language.SPANISH, text="es")

    AlphabetMethodAnalyzer.train(language=Language.ENGLISH, text=en_text)
    AlphabetMethodAnalyzer.train(language=Language.SPANISH, text=es_text)


def install_nltk_dependencies():
    nltk.download("punkt")
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


def analyze_document(file_path: str) -> Dict:
    text = FileReader(file_path).get_text()
    return {
        "word": WordFrequencyAnalyzer(text).execute(),
        "neural": NeuralMethodAnalyzer(text).execute(),
        "alphabet": AlphabetMethodAnalyzer(text).execute()
    }


def console_print(data: Dict):
    language_mapper: int = {
        Language.ENGLISH: "English",
        Language.SPANISH: "Spanish"
    }
    for method, result in data.items():
        print(f"{method}")
        for language, probability in result.items():
            print("\t",
                language_mapper.get(language),
                f"{int(probability*100)}%"
            )

def console_interface() -> Dict[Language, float]:
    document_path = input("Type path to the document: ")

    return analyze_document(f"tests_samples/{document_path}.txt")

if __name__ == "__main__":
    install_nltk_dependencies()
    train()
    console_print(console_interface())


# import spacy
# from spacy.language import Language
# from spacy_langdetect import LanguageDetector

# def get_lang_detector(nlp, name):
#     return LanguageDetector()

# en = spacy.load("en_core_web_sm")
# Language.factory("language_detector", func=get_lang_detector)
# Language.factory("language_detector", func=get_lang_detector)

# es.add_pipe('language_detector', last=True)
# en.add_pipe('language_detector', last=True)
# text = 'Русский.'
# doc = en(text)
# doc1 = es(text)
# print(doc._.language)
# print(doc1._.language)


# <spacy_langdetect.spacy_langdetect.LanguageDetector object at 0x7f9a32246250>
# <spacy.lang.en.English object at 0x7f2755492850>