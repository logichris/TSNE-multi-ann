# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: warn.undeclared=True
# cython: language_level=3
cimport numpy as np
import numpy as np
from cython.parallel import prange, parallel

import annoy.annoylib as annoylib


cdef class AnnoyIndex:
    cdef index
    cdef Py_ssize_t n_training_samples
    cdef Py_ssize_t n_dims

    def __init__(self, np.float64_t[:, :] data, str metric):
        cdef:
            Py_ssize_t n_samples = data.shape[0]
            Py_ssize_t n_dim = data.shape[1]

        self.n_training_samples = n_samples
        self.n_dims = n_dim
        self.index = annoylib.Annoy(n_dim, metric)
        self.add_items(data)

    cpdef add_items(self, np.float64_t[:, :] items):
        cdef:
            Py_ssize_t n_samples = items.shape[0]
            Py_ssize_t n_dim = items.shape[1]
            Py_ssize_t i

        for i in range(n_samples):
            self.index.add_item(i, items[i])

    cpdef add_item(self, Py_ssize_t idx, np.float64_t[:] item):
        self.index.add_item(idx, item)

    cpdef build(self, Py_ssize_t n_trees):
        self.index.build(n_trees)

    cpdef tuple query_train(self, Py_ssize_t k, Py_ssize_t search_k=-1, Py_ssize_t num_threads=1):
        cdef:
            Py_ssize_t idx, d, neighbor
            np.float64_t dist
            tuple result
            np.int64_t[:, :] indices = np.full((self.n_training_samples, k), -1, dtype=np.int64)
            np.float64_t[:, :] distances = np.full((self.n_training_samples, k), -1, dtype=np.float64)

        if num_threads < 1:
            num_threads = 1

        for idx in range(self.n_training_samples):
            result = self.index.get_nns_by_item(idx, k, search_k, include_distances=True)
            for d, (neighbor, dist) in enumerate(zip(*result)):
                indices[idx, d] = neighbor
                distances[idx, d] = dist

        return np.asarray(indices), np.asarray(distances)

    cpdef query(self, np.float64_t[:, :] data, Py_ssize_t k, Py_ssize_t search_k=-1, Py_ssize_t num_threads=1):
        cdef:
            Py_ssize_t n_samples = data.shape[0]
            Py_ssize_t idx, d, neighbor
            np.float64_t dist
            tuple result
            np.int64_t[:, :] indices = np.full((n_samples, k), -1, dtype=np.int64)
            np.float64_t[:, :] distances = np.full((n_samples, k), -1, dtype=np.float64)

        if num_threads < 1:
            num_threads = 1

        for idx in range(n_samples):
            result = self.index.get_nns_by_vector(data[idx], k, search_k, include_distances=True)
            for d, (neighbor, dist) in enumerate(zip(*result)):
                indices[idx, d] = neighbor
                distances[idx, d] = dist

        return np.asarray(indices), np.asarray(distances)
