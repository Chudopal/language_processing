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


def analyze_document(file_path):
    text = FileReader(file_path).get_text()

    print("word", WordFrequencyAnalyzer(text).execute())
    print("neural" ,NeuralMethodAnalyzer(text).execute())
    print("Alphabet" ,AlphabetMethodAnalyzer(text).execute())


def main():
    train()

if __name__ == "__main__":
    train()
    print("English")
    analyze_document("tests_samples/en/3_sample.txt")
    print("Spanish")
    analyze_document("tests_samples/es/3_sample.txt")
