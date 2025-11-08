"""
Seam Tokenizer Prototype

Provides a small prototype for encoding and decoding seam polylines as token sequences.
This is a lightweight utility (not dependent on UVs) demonstrating the token format used in
my Seam Tokenization Prototype write-up.

Tokens: S (start), E (end), V{idx} (base vertex), D{q} (direction code), L{b} (length bin)

Functions:
- build_dir_codebook(Q)
- quantize_dir(vec)
- encode_seam(vertex_indices, vertices, Q=26, B=8)
- decode_tokens(tokens, vertices, Q=26, B=8)
"""
import numpy as np
"""
Seam Tokenizer (updated)

Provides a compact, well-documented prototype for encoding seam polylines as token
sequences and decoding them back to 3D points. Improvements over the first draft:

- Use a Fibonacci-sphere codebook for more uniform direction sampling.
- Provide configurable length-binning modes (linear or log) and a helper to compute bins.
- Offer both human-readable token sequences and compact integer encodings (vocabulary mapping).
- Add a small CLI/demo and a unit-style self-test when run as __main__.

Token vocabulary (human-readable):
- S : start
- E : end
- V{idx} : base vertex (index into vertex array)
- D{d} : direction code (index into direction codebook)
- L{b} : length bin index

Compact encoding maps each token string to a small integer via an internal vocabulary
for compact storage or model-training experiments.
"""

from typing import List, Optional, Tuple, Dict, Sequence
import numpy as np

def build_dir_codebook(Q: int = 64) -> np.ndarray:
    """Build a near-uniform unit-direction codebook using a Fibonacci sphere.

    Args:
        Q: number of direction buckets.

    Returns:
        Array of shape (Q, 3) with unit vectors.
    """
    # Fibonacci sphere sampling
    indices = np.arange(0, Q, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/Q)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    dirs = np.vstack([x, y, z]).T
    # normalize to unit (numerical safety)
    dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    return dirs


def quantize_dir(vec: np.ndarray, codebook: np.ndarray) -> int:
    """Quantize a 3D direction vector to the nearest codebook index.

    Args:
        vec: 3-element array-like direction.
        codebook: (Q,3) unit vectors.

    Returns:
        index of the nearest codebook direction.
    """
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return 0
    v = v / n
    dots = codebook @ v
    return int(np.argmax(dots))


def compute_length_bins(steps: Sequence[float], B: int = 8, mode: str = 'linear') -> np.ndarray:
    """Compute length bin edges from observed step lengths.

    Args:
        steps: iterable of positive step lengths.
        B: number of bins.
        mode: 'linear' or 'log' (log spacing is useful when lengths vary widely).

    Returns:
        1D array of length B+1 with bin edges (ascending).
    """
    steps = np.asarray(list(steps), dtype=float)
    if len(steps) == 0:
        return np.linspace(0.0, 1.0, B+1)
    lo = 0.0
    hi = float(max(steps))
    if hi <= 0:
        return np.linspace(0.0, 1.0, B+1)
    if mode == 'log':
        # log spacing between small positive epsilon and hi
        eps = hi * 1e-6 if hi > 0 else 1e-6
        edges = np.logspace(np.log10(eps), np.log10(hi), num=B+1)
        edges[0] = 0.0
        return edges
    else:
        return np.linspace(0.0, hi, num=B+1)


def encode_seam(vertex_indices: Sequence[int],
                vertices: np.ndarray,
                Q: int = 64,
                B: int = 8,
                length_bins: Optional[np.ndarray] = None,
                length_mode: str = 'linear') -> Tuple[List[str], Dict[str, int]]:
    """Encode a seam (sequence of vertex indices) into a token sequence.

    Returns a tuple (tokens, vocab) where tokens is a list of human-readable tokens
    and vocab is a mapping token->int useful for compact encoding.
    """
    verts = np.asarray(vertices, dtype=float)
    if len(vertex_indices) == 0:
        return [], {}
    codebook = build_dir_codebook(Q)
    tokens: List[str] = []
    tokens.append('S')
    tokens.append(f'V{int(vertex_indices[0])}')

    # prepare length bins
    if length_bins is None:
        if len(vertex_indices) > 1:
            steps = []
            for i in range(len(vertex_indices)-1):
                a = verts[int(vertex_indices[i])]
                b = verts[int(vertex_indices[i+1])]
                steps.append(np.linalg.norm(b-a))
            length_bins = compute_length_bins(steps, B=B, mode=length_mode)
        else:
            length_bins = compute_length_bins([1.0], B=B, mode=length_mode)

    for i in range(len(vertex_indices)-1):
        a_idx = int(vertex_indices[i])
        b_idx = int(vertex_indices[i+1])
        a = verts[a_idx]
        b = verts[b_idx]
        delta = b - a
        d_idx = quantize_dir(delta, codebook)
        l = float(np.linalg.norm(delta))
        # digitize returns bin index in 1..B
        l_idx = int(np.digitize([l], length_bins, right=False)[0]) - 1
        l_idx = max(0, min(B-1, l_idx))
        tokens.append(f'D{d_idx}')
        tokens.append(f'L{l_idx}')

    tokens.append('E')

    # build a simple vocab mapping for compact integer encoding
    unique_tokens = sorted(set(tokens), key=lambda x: (x[0], x))
    vocab = {tok: i for i, tok in enumerate(unique_tokens)}
    return tokens, vocab


def tokens_to_ints(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    return [vocab[t] for t in tokens]


def ints_to_tokens(ints: Sequence[int], inv_vocab: Dict[int, str]) -> List[str]:
    return [inv_vocab[i] for i in ints]


def decode_tokens(tokens: Sequence[str], vertices: np.ndarray, Q: int = 64, B: int = 8,
                  length_bins: Optional[np.ndarray] = None) -> np.ndarray:
    """Decode a token sequence (human-readable) into reconstructed 3D points.

    Notes:
        - When length_bins is provided the L{b} tokens are mapped to the centre of the
          corresponding bin, otherwise a default length of 1.0 is used.
    """
    verts = np.asarray(vertices, dtype=float)
    codebook = build_dir_codebook(Q)
    pts = []
    i = 0
    pos = None
    while i < len(tokens):
        t = tokens[i]
        if t == 'S':
            i += 1
            if i >= len(tokens):
                break
            base_tok = tokens[i]
            if not base_tok.startswith('V'):
                raise ValueError('Expected V{idx} after S')
            base_idx = int(base_tok[1:])
            pos = verts[base_idx].astype(float).copy()
            pts.append(pos.copy())
            i += 1
        elif t.startswith('D'):
            d_idx = int(t[1:])
            i += 1
            if i >= len(tokens):
                break
            l_tok = tokens[i]
            if not l_tok.startswith('L'):
                raise ValueError('Expected L{b} after D{d}')
            l_idx = int(l_tok[1:])
            dir_vec = codebook[d_idx]
            if length_bins is None:
                length = 1.0
            else:
                a = length_bins[l_idx]
                b = length_bins[min(l_idx+1, len(length_bins)-1)]
                length = 0.5 * (a + b)
            pos = pos + dir_vec * length
            pts.append(pos.copy())
            i += 1
        elif t == 'E':
            i += 1
            break
        else:
            # unknown token: skip
            i += 1
    return np.array(pts)


def encode_seam_to_ints(vertex_indices: Sequence[int], vertices: np.ndarray, Q: int = 64, B: int = 8,
                        length_mode: str = 'linear') -> Tuple[List[int], Dict[int, str]]:
    """Convenience: encode seam and return compact integer sequence with inverse vocab.
    Returns (ints, inv_vocab) where inv_vocab maps int->token string.
    """
    tokens, vocab = encode_seam(vertex_indices, vertices, Q=Q, B=B, length_mode=length_mode)
    ints = tokens_to_ints(tokens, vocab)
    inv = {i: t for t, i in vocab.items()}
    return ints, inv


def _self_test():
    # synthetic seam and vertices
    verts = np.array([[0.,0.,0.],[1.,0.,0.],[2.,1.,0.],[3.,1.,1.]], dtype=float)
    seam = [0,1,2,3]
    tokens, vocab = encode_seam(seam, verts, Q=32, B=6)
    print('Tokens:', tokens)
    print('Vocab:', vocab)
    ints = tokens_to_ints(tokens, vocab)
    inv = {i:t for t,i in vocab.items()}
    print('Ints:', ints)
    toks2 = ints_to_tokens(ints, inv)
    recon = decode_tokens(toks2, verts, Q=32, B=6)
    print('Reconstructed points:\n', recon)


if __name__ == '__main__':
    _self_test()
