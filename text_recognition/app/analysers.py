import re
from abc import ABC, abstractmethod, abstractclassmethod
from string import punctuation

from models import Language
from typing import Dict, Set

import nltk
from langdetect import detect
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


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
                language: len(data - (data - set(self._text))
                    )/len(data)
            })

    def calculate_probapility(self) -> float:
        return len(self.train_data - set(self._text)
            )/len(self.train_data)

    @classmethod
    def train(cls, text: str, language: Language) -> None:
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
    def train(cls, text: str, language: Language) -> None:    
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
                probability = 1
            else:
                probability = 0
            self._language_probability.update({
                language: probability
            })

    @classmethod
    def train(cls, text: str, language: Language) -> None:
        cls.train_data.update({
            text: language
        })