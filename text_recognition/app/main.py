import os
from abc import ABC, abstractmethod, abstractclassmethod
from enum import Enum, auto
from typing import Dict, List, Set
from langdetect import detect
from string import punctuation

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


class Language(Enum):
    SPANISH = auto()
    ENGLISH = auto()


class FileReader():

    def __init__(self, path: str):
        self._path = path
        self._text: str = ""
        self._file_paths: List[str] = list()

    def get_text(self) -> str:
        self._find_files()
        self._read()
        return self._text

    def _find_files(self) -> str:
        try:
            self._file_paths = [
                self._path + file_path
                for file_path in
                os.listdir(self._path)
            ]
        except NotADirectoryError:
            self._file_paths = [self._path]

    def _read(self) -> None:
        for file_path in self._file_paths:
            with open(file_path) as file:
                self._text += file.read()


class Analyzer(ABC):

    punctuation: str =\
        punctuation + '\n\xa0«»\t'

    def __init__(self, text: str):
        self._text: str = text
        self._language_probability: Dict[
            Language, float] = dict()

    def execute(self) -> Language:
        self._text = self.prepare_data(
            self._text
        )
        self.analyze_text()
        return self._language_probability

    @classmethod
    def prepare_data(cls, text) -> str:
        text = text.lower()
        no_punct_text = "".join(
            [char for char in text
            if char not in cls.punctuation]
        )
        return re.sub(
            "\[[\d, \w]*\]|,|\.|!|:|#|'|\(|\)|\n|\[|\]",
            " ", no_punct_text)
        

    @abstractmethod
    def analyze_text(self) -> None:
        """Recogrizes text."""

    @abstractclassmethod
    def train(cls, text: str):
        """Trains model."""


class AlphabetMethodAnalyzer(Analyzer):

    train_data: Dict = dict()

    def analyze_text(self):
        for language, data in self.train_data.items():
            self._language_probability.update({
                language: len((data - set(self._text))
                    )/len(data)
            })

    def calculate_probapility(self) -> float:
        return len(self.train_data - set(self._text)
            )/len(self.train_data)

    @classmethod
    def train(cls, text: str, language: Enum) -> None:
        cls.train_data.update({
            language: set(cls.prepare_data(text))
        })


class WordFrequencyAnalyzer(Analyzer):

    train_data: Dict = dict()

    def analyze_text(self) -> None:
        for language, data in self.train_data.items():
            self._language_probability.update({
                language: len(data - (data - set(
                        self.find_base_form_set(self._text)))
                    )/len(data)
            })

    @classmethod
    def train(cls, text: str, language: Enum) -> None:    
        clear_text = cls.prepare_data(text)
        cls.train_data.update({
            language: cls.find_base_form_set(
                clear_text
            )
        })

    @classmethod
    def find_base_form_set(cls, text) -> Set[str]:
        tags = nltk.pos_tag(word_tokenize(text))
        return {
            WordNetLemmatizer().lemmatize(
                tag[0], cls.penn_to_wn(tag[1])
            ) for tag in tags
        }

    @staticmethod
    def penn_to_wn(tag) -> wn:
        result = wn.NOUN
        if tag in ['JJ', 'JJR', 'JJS']:
            result = wn.ADJ
        elif tag in ['RB', 'RBR', 'RBS']:
            result = wn.ADV
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            result = wn.VERB
        return result


class NeuralMethodAnalyzer(Analyzer):

    train_data: Dict = dict()

    def analyze_text(self):
        for _, language in self.train_data.items():
            if language == self.train_data.get(
                detect(self._text)):
                probability = 0.99
            else:
                probability = 0.01
            self._language_probability.update({
                language: probability
            })

    @classmethod
    def train(cls, text: str, language: Enum) -> None:
        cls.train_data.update({
            text: language
        })


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
    analyze_document("tests_samples/en/1_sample.txt")
    print("Spanish")
    analyze_document("tests_samples/es/1_sample.txt")
