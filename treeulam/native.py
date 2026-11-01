"""ctypes interface to the two exact native libraries in ../native.

prefix_search: candidate labels of one vertex by exact prefix branch-and-bound.
dp_native: min-plus tree-DP messages through a metric index, and batched
exact three-median costs. Partial results of an interrupted call are never used.
"""
from pathlib import Path
import ctypes as C
import sys
import time

import numpy as np

NATIVE_DIR = Path(__file__).resolve().parents[1] / 'native'
SUFFIX = '.dll' if sys.platform == 'win32' else '.so'
_libraries = {}


def _load(name):
    if name not in _libraries:
        path = NATIVE_DIR / (name + SUFFIX)
        if not path.is_file():
            raise FileNotFoundError(f'Missing native library {path}; run scripts/build_native.py')
        _libraries[name] = C.CDLL(str(path))
    return _libraries[name]


def _pointer(array, ctype):
    return array.ctypes.data_as(C.POINTER(ctype))


# ----------------------------------------------------------------------------- dp_native

def _dp():
    lib = _load('dp_native')
    if not getattr(lib, '_configured', False):
        p16, p32, p64 = C.POINTER(C.c_int16), C.POINTER(C.c_int32), C.POINTER(C.c_int64)
        lib.min_plus_index.argtypes = [C.c_int, C.c_int64, C.c_int64, p16, p16, p32, p64, p64,
                                       C.c_double, p64, p64, C.POINTER(C.c_uint64)]
        lib.min_plus_index.restype = C.c_int
        lib.median_costs.argtypes = [C.c_int, C.c_int64, p16, p16, p16, p32, C.c_double, p64]
        lib.median_costs.restype = C.c_int
        lib._configured = True
    return lib


def _check(code, timeout, cap):
    if code == 1:
        raise timeout()
    if code:
        raise cap('native DP allocation/error')


def min_plus(metric, Cu, rows, CA, FA, D, deadline, timeout, cap, limits=None, stats=None):
    """For each selected row x of Cu: min over a of FA[a] + d(x, CA[a]).

    The exact minimum is returned when it is at most the row's limit (default D);
    otherwise limit + 1 is returned. The second array holds the minimizing index or -1.
    """
    A = np.ascontiguousarray(Cu[rows], dtype=np.int16)
    B = np.ascontiguousarray(CA, dtype=np.int16)
    F = np.ascontiguousarray(FA, dtype=np.int64)
    w = np.ascontiguousarray(metric.w, dtype=np.int32)
    lim = np.full(len(A), D, dtype=np.int64) if limits is None else np.ascontiguousarray(limits, dtype=np.int64)
    assert lim.shape == (len(A),) and len(B) == len(F)
    out = np.empty(len(A), dtype=np.int64)
    arg = np.empty_like(out)
    counters = np.zeros(4, dtype=np.uint64)
    if not len(A):
        return out, arg
    remaining = deadline - time.time()
    if remaining <= 0:
        raise timeout()
    began = time.perf_counter()
    code = _dp().min_plus_index(A.shape[1], len(A), len(B), _pointer(A, C.c_int16), _pointer(B, C.c_int16),
                                _pointer(w, C.c_int32), _pointer(F, C.c_int64), _pointer(lim, C.c_int64),
                                remaining, _pointer(out, C.c_int64), _pointer(arg, C.c_int64),
                                _pointer(counters, C.c_uint64))
    if stats is not None:
        stats.append(dict(parent_rows=len(A), child_rows=len(B), build_distances=int(counters[0]),
                          query_distances=int(counters[1]), visited_nodes=int(counters[2]),
                          pruned_nodes=int(counters[3]), seconds=time.perf_counter() - began, status=code))
    _check(code, timeout, cap)
    return out, arg


def median_batch(a, b, rows, weights, deadline, timeout, cap):
    """Exact weighted three-median cost of (a, b, row) for every row."""
    a = np.ascontiguousarray(a, dtype=np.int16)
    b = np.ascontiguousarray(b, dtype=np.int16)
    rows = np.ascontiguousarray(rows, dtype=np.int16)
    w = np.ascontiguousarray(weights, dtype=np.int32)
    out = np.empty(len(rows), dtype=np.int64)
    remaining = deadline - time.time()
    if remaining <= 0:
        raise timeout()
    code = _dp().median_costs(len(a), len(rows), _pointer(a, C.c_int16), _pointer(b, C.c_int16),
                              _pointer(rows, C.c_int16), _pointer(w, C.c_int32), remaining,
                              _pointer(out, C.c_int64))
    _check(code, timeout, cap)
    return out


# ----------------------------------------------------------------------------- prefix_search

class _Result(C.Structure):
    _fields_ = [('status', C.c_int32), ('nodes', C.c_uint64), ('pruned', C.c_uint64),
                ('rows', C.c_uint64), ('elapsed', C.c_double),
                ('permutations', C.POINTER(C.c_int16)), ('distances', C.POINTER(C.c_int64)),
                ('owner', C.c_void_p)]


def _prefix():
    lib = _load('prefix_search')
    if not getattr(lib, '_configured', False):
        fn = lib.enumerate_prefix_domain
        fn.restype = C.POINTER(_Result)
        fn.argtypes = [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_int32), C.POINTER(C.c_int16),
                       C.POINTER(C.c_int64), C.c_int, C.c_int64, C.c_uint64, C.c_uint64, C.c_double,
                       C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_int32),
                       C.POINTER(C.c_int32), C.c_int]
        lib.release_prefix_domain.argtypes = [C.POINTER(_Result)]
        lib._configured = True
    return lib


def prefix_domain(labels, weights, branches, off_path, dm, D, cap, deadline, stats,
                  timeout, cap_error, byte_cap=256 * 1024 * 1024):
    """All weighted permutations whose summed branch lower bounds are at most D.

    labels: the distinct reduced leaf labels (rows of 0..n-1); branches: output of
    exact._branches for one vertex; off_path: constants from exact._off_path_bounds.
    Returns (rows, distances to every label). A timeout or cap raises instead.
    """
    refs = np.ascontiguousarray(labels, dtype=np.int16)
    q, n = refs.shape
    w = np.ascontiguousarray(weights, dtype=np.int32)
    terms = []
    for m, (u, labs, seqs) in enumerate(branches):
        for s in labs:
            terms.append(((m, s, -1, off_path.get((u, s), 0), 1), ()))
        for seq in seqs:
            inner = sum(dm[a][b] for a, b in zip(seq, seq[1:]))
            edges = tuple(sorted(min(a, b) * q + max(a, b) for a, b in zip(seq, seq[1:])))
            terms.append(((m, seq[0], seq[-1], inner, 2), edges))
    entries = sorted(set(terms))
    term_array = np.ascontiguousarray([t for t, _ in entries], dtype=np.int64).reshape(-1, 5)
    offsets, pairs = [0], []
    for _, e in entries:
        pairs.extend(e)
        offsets.append(len(pairs))
    term_offsets = np.array(offsets, dtype=np.int32)
    term_pairs = np.array(pairs, dtype=np.int32)
    no_tours = np.array([0], dtype=np.int32)
    no_pairs = np.array([], dtype=np.int32)
    if not 1 <= n < 32768 or q < 1 or len(w) != n or (w <= 0).any():
        raise ValueError('Unsupported alphabet or nonpositive weights')
    if not np.array_equal(np.sort(refs, axis=1), np.broadcast_to(np.arange(n), (q, n))):
        raise ValueError('References must be permutations of 0..n-1')
    if cap < 1 or byte_cap < 1 or not branches:
        raise ValueError('Positive candidate/output limits and nonempty branches required')
    if time.time() >= deadline:
        raise timeout()
    lib = _prefix()
    ptr = lib.enumerate_prefix_domain(
        n, q, len(branches), _pointer(w, C.c_int32), _pointer(refs, C.c_int16),
        _pointer(term_array, C.c_int64), len(term_array), D, cap, byte_cap, max(0, deadline - time.time()),
        _pointer(term_offsets, C.c_int32), _pointer(term_pairs, C.c_int32),
        _pointer(no_tours, C.c_int32), _pointer(no_pairs, C.c_int32), 0)
    if not ptr:
        raise cap_error('native result allocation failed')
    try:
        r = ptr.contents
        stats.append(dict(prefix_nodes=int(r.nodes), pruned_prefixes=int(r.pruned), candidates=int(r.rows),
                          search_seconds=float(r.elapsed), native_status=int(r.status),
                          output_bytes=int(r.rows) * (n * 2 + q * 8)))
        if r.status == 1:
            raise timeout()
        if r.status == 2:
            raise cap_error('prefix domain candidate or output-byte limit')
        if r.status:
            raise cap_error('native search allocation/error')
        if not r.rows:
            return np.empty((0, n), dtype=np.int16), np.empty((0, q), dtype=np.int64)
        # Copy before releasing native storage.
        rows = np.ctypeslib.as_array(r.permutations, shape=(r.rows * n,)).reshape(-1, n).copy()
        dist = np.ctypeslib.as_array(r.distances, shape=(r.rows * q,)).reshape(-1, q).copy()
        return rows, dist
    finally:
        lib.release_prefix_domain(ptr)
