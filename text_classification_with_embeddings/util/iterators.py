import collections.abc as abc

from text_classification_with_embeddings import LABEL_WORD_PREFIX


class SentenceIteratorFastTextFormat(abc.Iterator):
    """Iterator for iterating over sentences (documents) in fastText format with label words excluded"""

    def __init__(self, file_path: str):
        self._file_path = file_path

    def __len__(self):
        c = sum(1 for _ in self._f)
        self._f.seek(0)
        return c

    def __next__(self) -> str:
        try:
            line_nxt = next(self._f)
            return [word for word in line_nxt.split() if LABEL_WORD_PREFIX not in word]
        except StopIteration:
            self._f.seek(0)
            raise StopIteration

    def __enter__(self):
        self._f = open(self._file_path, 'r')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._f.close()
