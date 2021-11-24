import os
from typing import List


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
