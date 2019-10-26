import dash_table
import numpy as np
import pickle
from sklearn.decomposition import PCA
import zlib


class Embeddings:
    """Word Embeddings object. Stores data mapping word to index, index to
    word, and word to vector.
    """

    def __init__(self, w2i, w2vec):
        """

        Parameters
        ----------
        w2i: dict[str, int]
            Dictionary mapping word to its index in the vocabulary.
        w2vec: dict[str, np.array]
            Dictionary mapping word to its embedding.
        """
        self.w2i = w2i
        self.i2w = [w for w, i in sorted(self.w2i.items(), key=lambda x: x[1])]
        self.w2vec = w2vec
        self.vocab_size = len(w2i)
        self.dim = len(list(w2vec.values())[0])
        self.mat = self._build_embedding_matrix()
        self._mat_2d = None

    @classmethod
    def from_glove_file(cls, path, max_words=float('inf')):
        w2i = dict()
        w2vec = dict()
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i > max_words: break  # Faster testing
                word, *nums = line.strip().split()
                w2i[word] = i
                w2vec[word] = np.array(nums, dtype=float)
        return cls(w2i, w2vec)

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return zlib.decompress(data)

    def _build_embedding_matrix(self):
        mat = np.zeros((self.vocab_size, self.dim))
        for word, i in self.w2i.items():
            mat[i] = self.vec(word)
        return mat

    def vec(self, word):
        return self.w2vec.get(word.lower(), np.zeros(self.dim))

    def vec_2d(self, word):
        return self.mat_2d()[self[word]]

    def _distances(self, vec, distance='euclidean'):
        if distance == 'euclidean':
            return self.norm(self.mat - vec)
        elif distance == 'cosine':
            dists = self.cosine_distance(vec, self.mat)
        return dists

    def nearest_neighbors(self, word=None, vec=None, n=5, distance='euclidean',
                          digits=2):
        """Find the most similar words to a given word. User must pass in
        either a word OR a vector as the input.

        Parameters
        ----------
        word: str
            A word that must be in the vocabulary.
        vec: np.array
            A vector. This doesn't need to be in the vocabulary.
        n: int
            Number of neighbors to return.
        distance: str
            Distance method to use when computing nearest neighbors. One of
            ('euclidean', 'cosine').
        digits: int
            Digits to round output distances to.

        Returns
        -------
        dict[str, float]: Dictionary mapping word to distance.
        """
        assert bool(word is None) + bool(vec is None) == 1, \
            'Pass in either word or vec.'

        # If word is passed in, check that it's in vocab, then lookup vector.
        if word is not None and word not in self:
            return None
        if vec is None:
            vec = self.vec(word)

        dists = self._distances(vec, distance)
        idx = np.argsort(dists)[1:n+1]
        return {self.i2w[i]: round(dists[i], digits) for i in idx}

    def analogy(self, a, b, c, **kwargs):
        if not all(arg in self for arg in [a, b, c]):
            return None

        vec = self.vec(b) - self.vec(a) + self.vec(c)
        return self.nearest_neighbors(vec=vec, **kwargs)

    def mat_2d(self):
        """Compress the embedding matrix into 2 dimensions for human-readable
        plotting. Only perform computation once.

        Returns
        -------
        np.array: Embedding matrix after dimension reduction. Each vector has
            length 2.
        """
        # This may be a np.array so truth value is ambiguous.
        if self._mat_2d is not None:
            return self._mat_2d

        pca = PCA(n_components=2)
        self._mat_2d = pca.fit_transform(self.mat)
        return self._mat_2d

    @staticmethod
    def norm(vec):
        """Compute L2 norm of a vector. Euclidean distance between two vectors
        can be found by the operation norm(vec1 - vec2).

        Parameters
        ----------
        vec: np.array
            Input vector.

        Returns
        -------
        float: L2 norm of input vector.
        """
        return np.sqrt(np.sum(vec ** 2, axis=-1))

    def cosine_distance(self, vec1, vec2):
        """Compute cosine distance between two vectors.

        Parameters
        ----------
        vec1
        vec2

        Returns
        -------

        """
        return 1 - (np.sum(vec1 * vec2, axis=-1) /
                    (self.norm(vec1) * self.norm(vec2)))

    def save(self, path, verbose=True):
        """

        Parameters
        ----------
        path: str
            Path that object will be saved to.
        verbose

        Returns
        -------

        """
        with open(path, 'wb') as f:
            pickle.dump(zlib.compress(pickle.dumps(self)), f)
        if verbose:
            print(f'Embeddings object saved to {path}.')

    def __getitem__(self, word):
        return self.w2i.get(word.lower())

    def __len__(self):
        return self.vocab_size

    def __contains__(self, word):
        return word.lower() in self.w2i

    def __iter__(self):
        for word in self.w2i.keys():
            yield word


def get_empty_table_data(cols, nrows=5):
    return [{col: '' for col in cols} for i in range(nrows)]


def empty_table(cols, id_, nrows=5):
    """Generate a dash_table with empty data.

    Parameters
    ----------
    cols: list[str]
        Column names that will be used to define table.
    id_: str
        Identifier used for Dash callbacks so the table can be updated. This
        should be unique within the app.
    nrows: int
        Number of rows to display (default 5).

    Returns
    -------
    dash_table.DataTable
    """
    return dash_table.DataTable(columns=[{'name': col, 'id': col}
                                         for col in cols],
                                data=get_empty_table_data(cols, nrows),
                                style_table={'width': '50%'},
                                style_cell={'width': '50px',
                                            'minWidth': '50px',
                                            'maxWidth': '50px',
                                            'whiteSpace': 'normal'},
                                id=id_)


