from perseus_nlp_toolkit.reader import CapitainCorpusReader
from subprocess import run, PIPE
import os
from nltk.tokenize.api import TokenizerI

class TreeTaggerUtf8Tokenizer(TokenizerI):
    def __init__(self, path_to_script=None, abb_file=None, opts=None):
        self._script = os.path.join(path_to_script,"utf8-tokenize.perl") if path_to_script else "utf8-tokenize.perl"
        self._options = [o for o in opts] if opts else []
        self._abbreviation = ["-a", abb_file] if abb_file else []
        self.cmd = [self._script] + self._abbreviation + self._options

    def tokenize(self, s):
        p = run(self.cmd, stdout=PIPE, input=s, encoding="utf8")
        return p.stdout.split("\n")

