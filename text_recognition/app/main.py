from abc import ABC, abstractmethod
from enum import Emun, auto
from typing import Dict
from langdetect import detect

class Language(Emun):
    SPANISH = auto()
    ENGLISH = auto()


f = open('pushkin-metel.txt', "r", encoding="utf-8")
text = f.read()
text = text.lower()
import string
print(string.punctuation)
spec_chars = string.punctuation + '\n\xa0«»\t—…' 

def remove_chars_from_text(text, chars):
    return "".join([ch for ch in text if ch not in chars])

text = remove_chars_from_text(text, spec_chars)
text = remove_chars_from_text(text, string.digits)

from nltk import word_tokenize
text_tokens = word_tokenize(text)

import nltk
text = nltk.Text(text_tokens)

from nltk.probability import FreqDist
fdist = FreqDist(text)

from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")





class NeuralMethod:
    __slots__ = ('_path_to_file', '_content')

    LANGUAGES = {
        'en': 'ENGLISH',
        'es': 'SPANISH'
    }

    def __init__(self, path_to_file: str):
        self._path_to_file = path_to_file
        self.read_file()

    def read_file(self):
        with open(self._path_to_file, 'r', encoding='utf-8') as file:
            self._content = file.read()

    @property
    def get_result(self):
        return self.LANGUAGES.get(detect(self._content))


class Analyzer(ABC):

    def __init__(self, text: str):
        self._text = text

    @abstractmethod
    def execute(self):
        pass


class AlphabetMethodAnalyzer(Analyzer):

    def execute(self):
        pass


class NeuralMethodAnalyzer(Analyzer):

    def execute(self):
        return super().execute()


class WordFrequencyAnalyzer(Analyzer):

    def __init__(self, text):
        super().__init__(text)
        self._language_mapper: Dict[str, str] = {
            'en': Language.ENGLISH,
            'es': Language.SPANISH
        }

    def execute(self):
        return self._language_mapper.get(
            detect(self._text)
        )



def main():
    pass