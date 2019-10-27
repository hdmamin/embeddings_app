import dash_table
import numpy as np
import pickle
from sklearn.decomposition import PCA
import zlib

from htools import htimer


class Embeddings:
    """Word Embeddings object. Stores data mapping word to index, index to
    word, and word to vector.
    """

    def __init__(self, w2i, w2vec, mat=None, mat_2d=None):
        """
        Parameters
        ----------
        w2i: dict[str, int]
            Dictionary mapping word to its index in the vocabulary.
        w2vec: dict[str, np.array]
            Dictionary mapping word to its embedding.
        mat: np.array
            Matrix of embeddings, where row i corresponds to word i in vocab.
            If not passed in, it will be generated from w2vec and w2i.
        mat_2d: np.array
            Matrix output of PCA after compressing mat to vectors of length 2.
            If None, it will be computed from mat.
        """
        self.w2i = w2i
        self.i2w = [w for w, i in sorted(self.w2i.items(), key=lambda x: x[1])]
        self.w2vec = w2vec
        self.vocab_size = len(w2i)
        self.dim = len(list(w2vec.values())[0])
        self.mat = mat if mat is not None else self._build_embedding_matrix()
        self._mat_2d = mat_2d

    @classmethod
    @htimer
    def from_glove_file(cls, path, max_words=float('inf')):
        """Create a new Embeddings object from a raw csv file containing the
        GloVe vectors.

        Parameters
        ----------
        path: str
            Location of csv file containing GloVe vectors.
        max_words: int, float
            Set maximum number of words to read in from file. This can be used
            during development to reduce wait times when loading data.

        Returns
        -------
        Embeddings: Newly instantiated object.
        """
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
    @htimer
    def from_pickle(cls, path):
        """If an Embeddings object previously saved its data in a pickle file,
        loading it that way can avoid repeated computation.

        Parameters
        ----------
        path: str
            Location of pickle file.

        Returns
        -------
        Embeddings: Newly instantiated object using the data that was stored in
            the pickle file.
        """
        with open(path, 'rb') as f:
            data = zlib.decompress(pickle.load(f))
        return cls(**pickle.loads(data))

    def save(self, path, verbose=True):
        """Save data to a compressed pickle file. This reduces the amount of
        space needed for storage (the csv is much larger) and can let us
        avoid running PCA and building the embedding matrix again.

        Parameters
        ----------
        path: str
            Path that object will be saved to.
        verbose

        Returns
        -------
        None
        """
        data = dict(w2i=self.w2i,
                    w2vec=self.w2vec,
                    mat=self.mat,
                    mat_2d=self.mat_2d())
        with open(path, 'wb') as f:
            pickle.dump(zlib.compress(pickle.dumps(data)), f)
        if verbose:
            print(f'Embeddings object saved to {path}.')

    def _build_embedding_matrix(self):
        """Built a matrix of embeddings using the previously defined w2vec and
        w2idx.

        Returns
        -------
        np.array: Matrix of embeddings where each row corresponds to a single
            word.
        """
        mat = np.zeros((self.vocab_size, self.dim))
        for word, i in self.w2i.items():
            mat[i] = self.vec(word)
        return mat

    def vec(self, word):
        """Look up the embedding for a given word. Return None if not found.

        Parameters
        ----------
        word: str
            Input word to look up embedding for.

        Returns
        -------
        np.array: Embedding corresponding to the input word. If word not in
            vocab, return None.
        """
        return self.w2vec.get(word.lower(), None)

    def vec_2d(self, word):
        """Look up the compressed embedding for a word (PCA was used to shrink
        dimensionality to 2). Return None if the word is not present in vocab.

        Parameters
        ----------
        word: str
            Input work to look up.

        Returns
        -------
        np.array: Compressed embedding of length 2. None if not found.
        """
        idx = self.w2i.get(word)
        if idx is not None:
            return self.mat_2d()[idx]

    def _distances(self, vec, distance='euclidean'):
        """Find distance from an input vector to every other vector in the
        embedding matrix.

        Parameters
        ----------
        vec: np.array
            Vector for the input word.
        distance: str
            Specifies what distance metric to use for calculations.
            One of ('euclidean', 'cosine'). In a high-dimensional space such
            as this, cosine may be more appropriate, but euclidean is left as
            the default since a. this seems to be the expected distance metric
            in most cases and b. this may be more easily interpretable for
            the average person.

        Returns
        -------
        np.array: The i'th value corresponds to the distance to word i in the
            vocabulary.
        """
        if distance == 'euclidean':
            return self.norm(self.mat - vec)
        elif distance == 'cosine':
            dists = self.cosine_distance(vec, self.mat)
        return dists

    def nearest_neighbors(self, word, n=5, distance='euclidean', digits=2):
        """Find the most similar words to a given word. This wrapper to
        allows the user to pass in a word. To pass in a vector, use
        _nearest_neighbors().

        Parameters
        ----------
        word: str
            A word that must be in the vocabulary.
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
        # Error handling for words not in vocab.
        if word not in self:
            return None
        return self._nearest_neighbors(self.vec(word), n, distance, digits)

    def _nearest_neighbors(self, vec, n=5, distance='euclidean', digits=2):
        """Internal function behind nearest_neighbors(). This can be used if
        we want to pass in a vector instead of a word. For more details, see
        the wrapper method.

        Parameters
        ----------
        vec: np.array
        n: int
        distance: str
        digits: int

        Returns
        -------
        dict[str, float]: Dictionary mapping word to distance.
        """
        dists = self._distances(vec, distance)
        idx = np.argsort(dists)[1:n+1]
        return {self.i2w[i]: round(dists[i], digits) for i in idx}

    def analogy(self, a, b, c, **kwargs):
        """Fill in the analogy: a is to b as c is to ___.

        Parameters
        ----------
        a: str
            First word in analogy.
        b: str
            Second word in analogy.
        c: str
            Third word in analogy.
        kwargs

        Returns
        -------
        str: Word that would complete the analogy.
        """
        # If any words missing from vocab, arithmetic w/ None will throw error.
        try:
            vec = self.vec(b) - self.vec(a) + self.vec(c)
        except TypeError:
            return None
        return self._nearest_neighbors(vec, **kwargs)

    def cbow(self, *args):
        """Wrapper to _cbow() that allows us to pass in strings instead of
        vectors.

        Parameters
        ----------
        args: str
            Multiple words to average over.

        Returns
        -------
        np.array: Average of all input vectors. This will have the same
            embedding dimension as each input.
        """
        vecs = list(filter(lambda x: x is not None,
                           [self.vec(arg) for arg in args]))
        return self._cbow(*vecs)

    def _cbow(self, *args):
        """Internal helper for cbow(). Can also use this directly if you want
        to pass in vectors instead of words.

        Parameters
        ----------
        args: np.array
            Word vectors to average.

        Returns
        -------
        np.array: Average of all input vectors. This will have the same
            embedding dimension as each input.
        """
        return np.mean(args, axis=0)

    def cbow_neighbors(self, *args, **kwargs):
        """Wrapper to cbow(). This lets us pass in words, compute their
        average embedding, then return the words nearest this embedding.

        Parameters
        ----------
        args: str
            Input words to average over.
        kwargs: n (int), distance (str), digits (int)
            See _nearest_neighbors() for details.

        Returns
        -------
        dict[str, float]: Dictionary mapping word to distance from the average
            of the input words' vectors.
        """
        print(args)
        vec_avg = self.cbow(*args)
        print(vec_avg)
        # TODO: Maybe need error checking here.
        return self._nearest_neighbors(vec_avg, **kwargs)

    def mat_2d(self):
        """Compress the embedding matrix into 2 dimensions for human-readable
        plotting. Only perform computation once.

        Returns
        -------
        np.array: Embedding matrix after dimension reduction. Each vector has
            length 2.
        """
        # This may be a np.array so truth value is ambiguous.
        if self._mat_2d is None:
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
        float: Cosine distance between vec1 and vec2.
        """
        return 1 - (np.sum(vec1 * vec2, axis=-1) /
                    (self.norm(vec1) * self.norm(vec2)))

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


