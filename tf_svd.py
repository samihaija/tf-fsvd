import numpy as np
import scipy.sparse
import tensorflow as tf


class ProductFn:
  """Abstract class. Instances can be passed to function `fsvd`.

  An intance of (a concrete implementation of) this class would hold an implicit
  matrix `M`, such that, this class is able to multiply it with another matrix
  `m` (by implementing function `dot`).

  Attribute `T` should evaluate to a `ProductFn` with implicit matrix being
  transpose of `M`.

  `shape` attribute must evaluate to shape of `M`
  """

  def dot(self, m):
    raise NotImplementedError(
      'dot: must be able to multiply (implicit) matrix by another matrix `m`.')


  @property
  def T(self):
    raise NotImplementedError(
      'T: must return instance of ProductFn that is transpose of this one.')

  @property
  def shape(self):
    raise NotImplementedError(
      'shape: must return shape of implicit matrix.')


## Functional TF implementation of Truncated Singular Value Decomposition
# The algorithm is based on Halko et al 2009 and their recommendations, with
# some ideas adopted from code of scikit-learn.

def fsvd(fn, k, n_redundancy=None, n_iter=10):
  """Functional TF Randomized SVD based on Halko et al 2009

  Args:
    fn: Instance of a class implementing ProductFn. Should hold implicit matrix
      `M` with (arbitrary) shape. Then, it must be that `fn.shape == (r, c)`,
      and `fn.dot(M1)` where `M1` has shape `(c, s)` must return `M @ M1` with
      shape `(r, s)`. Further, `fn.T.dot(M2)` where M2 has shape `(r, h)` must
      return `M @ M2` with shape `(c, h)`.
    k: rank of decomposition. Returns (approximate) top-k singular values in S
      and their corresponding left- and right- singular vectors in U, V, such
      that, `tf.matmul(U * S, V, transpose_b=True)` is the best rank-k
      approximation of matrix `M` (implicitly) stored in `fn`.
    n_redundancy: rank of "randomized" decomposition of Halko. The analysis of
      Halko provides that if n_redundancy == k, then the rank-k SVD approximation
      is, in expectation, no worse (in frobenius norm) than twice of the "true"
      rank-k SVD compared to the (implicit) matrix represented by fn.
      However, n_redundancy == k is too slow when k is large. Default sets it
      to min(k, 30).
    n_iter: Number of iterations. >=4 gives good results (with 4 passes over the
      data). We set to 10 (slower than 4) to ensure close approximation accuracy.
      The error decays exponentially with n_iter.
  Returns:
    U, s, V, s.t. tf.matmul(U*s, V, transpose_b=True) is a rank-k approximation
    of fn.
  """

  if n_redundancy is None:
    n_redundancy = min(k, 30)
  n_random = k + n_redundancy
  n_samples, n_features = fn.shape
  transpose = n_samples < n_features
  if transpose:
    # This is faster
    fn = fn.T

  Q = tf.random.normal(shape=(fn.shape[1], n_random))
  for i in range(n_iter):
    # Halko says it is more accurate (but slower) to do QR decomposition here.
    # TODO: Provide a faster (but less accurate) version.
    Q, _ = tf.linalg.qr(fn.dot(Q))
    Q, _ = tf.linalg.qr(fn.T.dot(Q))
  
  Q, _ = tf.linalg.qr(fn.dot(Q))

  B = tf.transpose(fn.T.dot(Q))
  s, Uhat, V = tf.linalg.svd(B)
  del B
  U = tf.matmul(Q, Uhat)

  U, V = _sign_correction(u=U, v=V, u_based_decision=not transpose)

  if transpose:
    return V[:, :k], s[:k], U[:, :k]
  else:
    return U[:, :k], s[:k], V[:, :k]


def _sign_correction(u, v, u_based_decision=True):
    M = u if u_based_decision else v
    max_abs_cols = tf.argmax(tf.abs(M), axis=0)
    signs = tf.sign(tf.gather_nd(M, tf.stack([max_abs_cols, tf.range(M.shape[1], dtype=tf.int64)], axis=1)))
    
    return u*signs, v*signs

# End of: Functional TF implementation of Truncated Singular Value Decomposition
## 



#### ProductFn implementations.


class SparseMatrixPF(ProductFn):
  """The "implicit" matrix comes directly from a scipy.sparse.csr_matrix

  This is the most basic version: i.e., this really only extends TensorFlow to
  run "sparse SVD" on a matrix. The given `scipy.sparse.csr_matrix` will be
  converted to `tf.sparse.SparseTensor`.
  """

  def __init__(self, csr_mat=None, precomputed_tfs=None, T=None):
    """Constructs matrix from csr_mat (or alternatively, tf.sparse.tensor).

    Args:
      csr_mat: instance of scipy.sparse.csr_mat (or any other sparse matrix
        class). This matrix will only be read once and converted to
        tf.sparse.SparseTensor.
      precomputed_tfs: (optional) matrix (2D) instance of tf.sparse.SparseTensor.
        if not given, will be initialized from `csr_mat`.
      T: (do not provide) if given, must be instance of ProductFn with implicit
        matrix as the transpose of this one. If not provided (recommended) it
        will be automatically (lazily) computed.
    """
    if precomputed_tfs is None and csr_mat is None:
      raise ValueError('Require at least one of csr_mat or precomputed_tfs')
    if precomputed_tfs is None:
      rows, cols = csr_mat.nonzero()
      values = np.array(csr_mat[rows, cols], dtype='float32')[0]
      precomputed_tfs = tf.sparse.SparseTensor(
        tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1),
        values,
        csr_mat.shape)
   
    self._shape = precomputed_tfs.shape
    self.csr_mat = csr_mat
    self.tfs = precomputed_tfs  # tensorflow sparse tensor.
    self._t = T

  def dot(self, v):
    return tf.sparse.sparse_dense_matmul(self.tfs, v)

  @property
  def T(self):
    """Returns ProductFn with implicit matrix being transpose of this one."""
    if self._t is None:
      self._t = SparseMatrixPF(
            self.csr_mat.T if self.csr_mat is not None else None,
            precomputed_tfs=tf.sparse.transpose(self.tfs),
            T=self)
    
    return self._t

  @property
  def shape(self):
    return self._shape


class BlockWisePF(ProductFn):
  """Product that concatenates, column-wise, one or more (implicit) matrices.

  Constructor takes one or more ProductFn instances. All of which must contain
  the same number of rows (e.g., = r) but can have different number of columns
  (e.g., c1, c2, c3, ...). As expected, the resulting shape will have the same
  number of rows as the input matrices and the number of columns will is the sum
  of number of columns of input (shape = (r, c1+c2+c3+...)).
  """

  def __init__(self, fns, T=None, concat_axis=1):
    """Concatenate (implicit) matrices stored in `fns`, column-wise.

    Args:
      fns: list. Each entry must be an instance of class implementing ProductFn.
      T: (do not provide) if given, must be instance of ProductFn with implicit
        matrix as the transpose of this one. If not provided (recommended) it
        will be automatically (lazily) computed.
      concat_axis: fixed to 1 (i.e. concatenates column-wise).
    """
    self.fns = fns
    self._t = T
    self.concat_axis = concat_axis

  @property
  def shape(self):
    size_other_axis = self.fns[0].shape[1 - self.concat_axis]
    for fn in self.fns[1:]:
      assert fn.shape[1 - self.concat_axis] == size_other_axis
    total = sum([fn.shape[self.concat_axis] for fn in self.fns])
    myshape = [0, 0]
    myshape[self.concat_axis] = total
    myshape[1 - self.concat_axis] = size_other_axis
    return tuple(myshape)
  
  def dot(self, v):
    if self.concat_axis == 0:
      dots = [fn.dot(v) for fn in self.fns]
      return tf.concat(dots, axis=self.concat_axis)
    else:
      dots = []
      offset = 0
      for fn in self.fns:
        fn_columns = fn.shape[1]
        dots.append(fn.dot(v[offset:offset+fn_columns]))
        offset += fn_columns
      return tf.reduce_sum(dots, axis=0)

  @property
  def T(self):
    """Returns ProductFn with implicit matrix being transpose of this one."""
    if self._t is None:
      fns_T = [fn.T for fn in self.fns]
      self._t = BlockWisePF(fns_T, T=self, concat_axis=1 - self.concat_axis)
    return self._t


class DenseMatrixPF(ProductFn):
  """Product function where implicit matrix is Dense tensor.

  On its own, this is not needed as one could just run tf.linalg.svd directly
  on the implicit matrix. However, this is useful when a dense matrix to be
  concatenated (column-wise) next to SparseMatrix (or any other implicit matrix)
  implementing ProductFn.
  """

  def __init__(self, m, T=None):
    """
    Args:
      m: tf.Tensor (dense 2d matrix). This will be the "implicit" matrix.
      T: (do not provide) if given, must be instance of ProductFn with implicit
        matrix as the transpose of this one. If not provided (recommended) it
        will be automatically (lazily) computed.
    """
    self.m = m
    self._t = T

  def dot(self, v):
    return tf.matmul(self.m, v)
  
  @property
  def shape(self):
    return self.m.shape

  @property
  def T(self):
    """Returns ProductFn with implicit matrix being transpose of this one."""
    if self._t is None:
      self._t = DenseMatrixPF(tf.transpose(self.m), T=self)
    return self._t


class WYSDeepWalkPF(ProductFn):
  """ProductFn for matrix approximating Watch Your Step derivation of DeepWalk.
  """
  
  def __init__(self, csr_adj, window=10, mult_degrees=False,
               Q=None, neg_sample_coef=None,
               tfs_unnormalized=None, tfs_normalized=None, tfs_degrees=None,
               T=None):
    """Constructs (implicit) matrix as approximating WYS derivation of DeepWalk.

    The implicit matrix looks like:

      M = \sum_i (Tr)^i q_i

    where q_i is entry in vector `Q`.
    
    Optionally (following WYS codebase):

      M := M * degrees  # only if `mult_degrees` is set.

    Args:
      csr_adj: Binary adjacency matrix as scipy.sparse.csr_mat (or any other
        scipy.sparse matrix class). Read only once and converted to tensorflow.
      window: Context window size (hyperparameter is C in WYS & our paper).
      mult_degrees: If set, the implicit matrix will be multipled by diagonal
        matrix of node degrees. Effectively, this starts a number of walks
        proportional from each node proportional to its degree.
      Q: Context distribution. Vector of size `C=window` that will be used for
        looking up q_1, ..., q_C. Entries should be positive but need not add
        to one. In paper, the entries are referred to c_1, ... c_C.
      neg_sample_coef: Scalar coefficient of the `(1-A)` term in implicit matrix
        `M`.
      tfs_unnormalized: Optional. If given, it must be a 2D matrix of type
         `tf.sparse.Tensor` containing the adjacency matrix (i.e. must equal
         to csr_adj, but with type tf). If not given, it will be constructed
         from `csr_adj`.
      tfs_normalized: Optional. If given, it must be a 2D matrix of type
        `tf.sparse.Tensor` containing the row-normalized transition matrix i.e.
        each row should sum to one. If not given, it will be computed.
      tfs_degrees: Optional. It will be computed if tfs_normalized is to be
        computed. If given, it must be a tf.sparse.SparseTensor diagonal matrix
        containing node degrees along the diagonal.
    """
    self.mult_degrees = mult_degrees
    self.neg_sample_coef = neg_sample_coef
    self._t = T  # Transpose
    self.window = window
    self.csr_mat = csr_adj
    if Q is None:
      Q = window - tf.range(window, dtype='float32')  # Default of deepwalk per WYS
    self.Q = Q

    rows, cols = csr_adj.nonzero()
    n, _ = csr_adj.shape
    if tfs_unnormalized is None:
      tfs_unnormalized = tf.sparse.SparseTensor(
        tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1),
        tf.ones(len(rows), dtype=tf.float32),
        (n, n))
    self.tfs_unnormalized = tfs_unnormalized

    if tfs_normalized is None:
      # Normalize
      degrees = np.array(csr_adj.sum(axis=1))[:, 0]
      degrees = np.clip(degrees, 1, None)
      inv_degrees = scipy.sparse.diags(1.0/degrees)
      csr_normalized = inv_degrees.dot(csr_adj)
      tfs_normalized = tf.sparse.SparseTensor(
        tf.stack([np.array(rows, dtype='int64'), np.array(cols, dtype='int64')], axis=1),
        np.array(csr_normalized[rows, cols], dtype='float32')[0],
        (n, n))

      tfs_degrees = tf.sparse.SparseTensor(
        tf.stack([tf.range(n, dtype=tf.int64)]*2, axis=1),
        np.array(degrees, dtype='float32'),
        (n, n))
    self.tfs_normalized = tfs_normalized
    self.tfs_degrees = tfs_degrees

  @property
  def T(self):
    """Returns ProductFn with implicit matrix being transpose of this one."""
    if self._t is not None:
      return self._t
    
    self._t = WYSDeepWalkPF(
        self.csr_mat.T,
        window=self.window,
        mult_degrees=self.mult_degrees,
        tfs_normalized=tf.sparse.transpose(self.tfs_normalized),
        tfs_unnormalized=tf.sparse.transpose(self.tfs_unnormalized),
        tfs_degrees=self.tfs_degrees,
        Q=self.Q,
        T=self,
        neg_sample_coef=self.neg_sample_coef)

    return self._t

  @property
  def shape(self):
    return self.csr_mat.shape


  def dot(self, v):
    product = v
    if self.mult_degrees:
      product = tf.sparse.sparse_dense_matmul(self.tfs_degrees, product)  # Can be commented too
    geo_sum = 0
    for i in range(self.window):
      product = tf.sparse.sparse_dense_matmul(self.tfs_normalized, product)
      geo_sum += self.Q[i] * product
    
    row_ones = tf.ones([1, self.csr_mat.shape[0]], dtype=tf.float32)
    neg_part = -tf.matmul(row_ones, tf.matmul(row_ones, v), transpose_a=True) + tf.sparse.sparse_dense_matmul(self.tfs_unnormalized, v)

    return geo_sum + self.neg_sample_coef * neg_part


def test_rsvdf():
  import scipy.sparse as sp
  M = sp.csr_matrix((50, 100))
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      if (i+j) % 2 == 0:
        M[i, j] = i + j

  u,s,v = fsvd(SparseMatrixPF(M), 4)
  assert np.all(np.abs(M.todense() - tf.matmul(u*s, v, transpose_b=True).numpy()) < 1e-3)

  M = M.T
  u,s,v = fsvd(SparseMatrixPF(M), 4)
  assert np.all(np.abs(M.todense() - tf.matmul(u*s, v, transpose_b=True).numpy()) < 1e-3)
  
  print('Test passes.')

if __name__ == '__main__':
  test_rsvdf()
