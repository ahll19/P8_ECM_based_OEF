import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft


def td2fd(xs):
    # input: N (even) real numbers
    # output: 1 + N // 2 complex numbers (the imaginary part of last number is zero)
    # or
    # input: N (odd) real numbers
    # output: 1 + N // 2 complex numbers (the imaginary part of last number is not zero)
    # complexity: O(n·logn)

    # wrapper
    original_dimension = len(xs.shape)
    if original_dimension < 2:
        xs = xs[np.newaxis, :]

    zs = fft(xs, axis=-1) * (2 / xs.shape[-1])
    zs[:, 0] /= 2
    if xs.shape[-1] % 2 == 0:
        zs[:, xs.shape[-1] // 2] /= 2

    # recover
    if original_dimension == 1:
        return zs[0, :xs.shape[-1] // 2 + 1]

    return zs[:, :xs.shape[-1] // 2 + 1]


def fd2td(zs, is_even=True):
    # input: 1 + N complex numbers (the imaginary part of last number is zero)
    # output: N * 2 (even) real numbers
    # or
    # input: 1 + N complex numbers (the imaginary part of last number is not zero)
    # output: 1 + N * 2 (odd) real numbers
    # complexity: O(n·logn)

    # wrapper
    original_dimension = len(zs.shape)
    if original_dimension < 2:
        zs = zs[np.newaxis, :]

    if is_even:
        zs = np.hstack((zs, np.flip(zs[:, 1:-1].conj(), axis=-1)))  # (N + 1) + (N - 1) = 2N
        zs *= zs.shape[1] / 2
        zs[:, 0] *= 2
        zs[:, zs.shape[1] // 2] *= 2
        xs = ifft(zs, axis=-1).real
    else:
        zs = np.hstack((zs, np.flip(zs[:, 1:].conj(), axis=-1)))  # (N + 1) + N = 2N + 1
        zs *= zs.shape[1] / 2
        zs[:, 0] *= 2
        xs = ifft(zs, axis=-1).real

    return xs[0] if original_dimension == 1 else xs


def naive_td2fd(xs):
    # for illustrative purpose
    # equivalent function to `td2fd`, but `only support 1d array` and `far less efficient`.
    # complexity: O(n^2)

    j = complex(0, 1)
    T = len(xs)
    zs = np.zeros(1 + T // 2, dtype=complex)
    for k in range(1 + T // 2):
        zs[k] = sum(xs[t] * np.exp(-j * 2 * np.pi / T * k * t) for t in range(T)) / T
        if not (k == 0 or (k == T // 2 and T % 2 == 0)):
            zs[k] *= 2
    return zs


def naive_fd2td(zs):
    # for illustrative purpose
    # equivalent function to `fd2td`, but `only support 1d array` and `far less efficient`.
    # complexity: O(n^2)

    N = len(zs) - 1
    T = 2 * N + (0 if abs(zs[-1].imag) < 1e-8 else 1)
    ts = np.zeros(T)
    for t in range(T):
        ts[t] = sum(zs[k].real * np.cos(2 * np.pi / T * k * t) - zs[k].imag * np.sin(2 * np.pi / T * k * t)
                    for k in range(N + 1))
    return ts


def get_DFT_matrix(T):
    # zs == td2fd(xs) == get_DFT_matrix(len(xs)) @ xs

    N = 1 + T // 2
    j = complex(0, 1)
    omega = np.exp(-j * 2 * np.pi / T)

    dft_matrix = np.ones((N, T), dtype=complex) / T
    for row in range(1, N):
        for col in range(T):
            dft_matrix[row, col] *= 2 * omega ** (row * col)

    if not T & 1:
        for col in range(T):
            dft_matrix[-1, col] /= 2

    return dft_matrix


def get_IDFT_matrix(T):
    # xs == fd2td(zs) == (get_IDFT_matrix(len(xs)) @ zs).real

    N = 1 + T // 2
    j = complex(0, 1)
    omega = np.exp(j * 2 * np.pi / T)

    idft_matrix = np.ones((N, T), dtype=complex)
    for row in range(1, N):
        for col in range(T):
            idft_matrix[row, col] = omega ** (row * col)

    return idft_matrix.T


def get_standard_IDFT_matrix(N):
    j = complex(0, 1)
    omega = np.exp(j * 2 * np.pi / N)

    ift = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for k in range(N):
            ift[i, k] = omega ** (i * k)

    return ift


def get_IFFT_matrices(N):
    def get_block_N(n):
        assert 2 ** np.round(np.log2(n)) == n
        if n == 2:
            return np.asarray([[1, 1], [1, -1]], dtype=complex)
        mat = np.zeros((n, n), dtype=complex)
        w = np.exp(j * 2 * np.pi / n)
        half = n // 2
        for i in range(half):
            mat[i, i] = 1
            mat[i, i + half] = w ** i
            mat[i + half, i] = 1
            mat[i + half, i + half] = -w ** i
        return mat

    def diag_merge(*mats):
        n = sum([mat.shape[0] for mat in mats])
        k = 0
        merged_mat = np.zeros((n, n), dtype=complex)
        for mat in mats:
            merged_mat[k:k + mat.shape[0], k:k + mat.shape[0]] = mat
            k += mat.shape[0]
        return merged_mat

    def rearrange_columns(n):
        def swap(arr, left, right):
            tmp = np.hstack((arr[left:right:2], arr[left + 1:right:2]))
            arr[left:right] = tmp

        cols = np.arange(n)
        interval = n
        for i in range(round(np.log2(n)) - 1):
            for j in range(n // interval):
                swap(cols, j * interval, j * interval + interval)
            interval //= 2
        return cols

    j = complex(0, 1)
    assert 2 ** int(np.log2(N)) == N  # more general cases to be addressed

    iffts = []
    n = N
    while n > 1:
        block = get_block_N(n)
        iffts.append(diag_merge(*[block] * (N // n)))
        n //= 2
    iffts[-1] = iffts[-1][:, rearrange_columns(N)]

    return iffts


if __name__ == '__main__':
    # unit test
    np.set_printoptions(linewidth=10000)

    # test: time domain to frequency domain
    x_td = np.random.random((10, 100))  # each row represents a time series
    print(td2fd(x_td)[5].round(9))
    print(naive_td2fd(x_td[5]).round(9))
    print((get_DFT_matrix(len(x_td[5])) @ x_td[5]).round(9))

    # test: frequency domain to time domain
    z_fd = td2fd(x_td)
    print(x_td[8])
    print(fd2td(z_fd[8]))
    print(naive_fd2td(z_fd[8]))
    print((get_IDFT_matrix(len(x_td[8])) @ z_fd[8]).real)

    # test: IFFT
    IFFTs = get_IFFT_matrices(32)
    IDFT = get_standard_IDFT_matrix(32)
    from functools import reduce
    print(f"max error = {abs(reduce(np.matmul, IFFTs) - IDFT).max()}")

    print("test over.")
