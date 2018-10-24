from nltk.corpus import ConllCorpusReader
from nltk.util import LazyMap, LazyConcatenation
from nltk.corpus.reader.util import read_blankline_block, concat, StreamBackedCorpusView


class CommentedConllCorpusReader(ConllCorpusReader):
    '''
    A corpus reader for conll files with comments between sentences
    '''

    def __init__(self, root, fileids, columntypes):
        super().__init__(root, fileids, columntypes)

    def _read_grid_block(self, stream):
        grids = []
        for block in read_blankline_block(stream):
            block = block.strip()
            if not block: continue

            grid = [line.split() for line in block.split('\n') if line[0] != "#"]

            if not grid: continue

            # If there's a docstart row, then discard. ([xx] eventually it
            # would be good to actually use it)
            if grid[0][self._colmap.get('words', 0)] == '-DOCSTART-':
                del grid[0]

            # Check that the grid is consistent.
            for row in grid:
                if len(row) != len(grid[0]):
                    raise ValueError('Inconsistent number of columns:\n%s' % block)
            grids.append(grid)
        return grids


class WebAnnoTsvCorpusReader(CommentedConllCorpusReader):

    WORDS = 'words'  #: column type for words
    POS = 'pos'  #: column type for part-of-speech tags
    TREE = 'tree'  #: column type for parse trees
    CHUNK = 'chunk'  #: column type for chunk structures
    NE = 'ne'  #: column type for named entities
    SRL = 'srl'  #: column type for semantic role labels
    COREF = 'coref' #: column type for coreference resolution
    LEMMA = 'lemma' #: column type for lemma
    OFFSET = 'offset'
    CUSTOM = 'custom' #: column type for custom column in WebAnno
    IGNORE = 'ignore'  #: column type for column that should be ignored

    COLUMN_TYPES = (WORDS, POS, LEMMA, TREE, CHUNK, NE, SRL, COREF, OFFSET, CUSTOM, IGNORE)

    def __init__(self, root, fileids, columntypes):
        super().__init__(root, fileids, ["ignore"])
        for columntype in columntypes:
            if columntype not in self.COLUMN_TYPES:
                raise ValueError('Bad column type %r' % columntype)

        #self.COLUMN_TYPES = columntypes
        self._colmap = dict((c, i) for (i, c) in enumerate(columntypes))
        self.columns = list(self._colmap.keys())

    def iob_words(self, fileids=None, columns=None, convert2iob=[]):
        """
        :return: a list of word/tag/IOB tuples
        :rtype: list(tuple)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        if columns is None:
            columns = [c for c in self.columns if c != "ignore"]

        self._require(*columns)

        def get_iob_words(grid):
            return self._get_converted_iob_words(grid, columns, convert2iob=convert2iob)

        return LazyConcatenation(LazyMap(get_iob_words, self._grids(fileids)))

    def iob_sents(self, fileids=None, columns=None, convert2iob=[]):
        """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        if columns is None:
            columns = [c for c in self.columns if c != "ignore"]

        self._require(*columns)

        def get_iob_words(grid):
            return self._get_converted_iob_words(grid, columns, convert2iob=convert2iob)

        return LazyMap(get_iob_words, self._grids(fileids))

    def _get_converted_iob_words(self, grid, columns, convert2iob = []):
        #pos_tags = self._get_column(grid, self._colmap['pos'])

        #return list(
        #    zip(*[self._get_column(grid, self._colmap[c]) for c in columns])
        # )
        words = []
        for c in columns:
            if c in convert2iob:
                words.append(self._tsv2iob(self._get_column(grid, self._colmap[c])))
            else:
                words.append(self._get_column(grid, self._colmap[c]))

        return list(zip(*words))


    def _tsv2iob(self, tags):

        def normalize_ne_tag(tag):
            """
            WebAnno's TSV format uses square brackets to represent entities
            made up of several tokens. This functions returns just the entity
            tag.
            """
            return tag.split("[")[0] if "[" in tag else tag

        converted_tags = []

        for i, tok in enumerate(tags):
            prev_token = tags[i - 1] if tags[-1] != "_" else "O"
            ne = tok if tok != "_" else "O"
            if ne != "O" and i == 0:
                conv = "B-%s" % normalize_ne_tag(ne)
            elif ne != "O" and prev_token == "O":
                conv = "B-%s" % normalize_ne_tag(ne)
            elif ne != "O" and prev_token != "O" and ne != prev_token:
                conv = "B-%s" % normalize_ne_tag(ne)
            elif ne != "O" and prev_token != "O" and ne == prev_token:
                conv = "I-%s" % normalize_ne_tag(ne)
            elif ne != "O" and ne == prev_token:
                conv = "I-%s" % normalize_ne_tag(ne)
            else:
                conv = normalize_ne_tag(ne)

            converted_tags.append(conv)

        return converted_tags

