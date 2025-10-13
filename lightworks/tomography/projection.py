# Copyright 2025 - 2025 Aegiq Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

# NOTE: This file is auto-documented, so any functions not intended to be public
# should begin with _

T = TypeVar("T", bound=np.generic)


def project_density_to_physical(
    rho: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """
    Takes a provided density matrix and projects into to a physical space using
    the algorithm from https://doi.org/10.48550/arXiv.1106.5458. It ensures:

        1. The matrix is Hermitian
        2. The trace of the matrix is 1.
        3. All eigenvalues of the matrix are positive, if not the matrix is
           projected so this is the case.

    Args:

        rho (np.ndarray) : The matrix which is to be projected.

    Returns:

        np.ndarray : The calculated density matrix, which meets the required
            conditions to be physical.

    """
    dim = rho.shape[0]
    # Ensure matrix is hermitian
    rho = (rho + rho.conj().T) / 2
    # And that the trace is 1
    rho /= np.trace(rho)
    # Then find eigenvalues and eigenvectors
    eigvals, eigvecs = _find_eigen_data(rho)
    # Don't do anything else if the eigenvalues are all positive
    if min(eigvals) >= 0:
        return rho
    # Otherwise apply algorithm - requires eigenvalues are sorted from largest
    # to smallest so invert these first
    eigvals = np.flip(eigvals)
    i = dim
    acc = 0.0
    while eigvals[i - 1] + acc / i < 0:
        acc += eigvals[i - 1]
        i -= 1
    for j in range(dim):
        if j < i:
            eigvals[j] += acc / i
        else:
            eigvals[j] = 0
    # Return eigenvalues to the correct order
    eigvals = np.flip(eigvals)
    # Reconstruct the matrix and return
    return eigvecs @ np.diag(eigvals) @ np.conj(eigvecs.T)


def project_choi_to_physical(
    choi: NDArray[np.complex128], max_iter: int = 1000
) -> NDArray[np.complex128]:
    """
    Performs the CPTP algorithm from https://arxiv.org/abs/1803.10062, ensuring
    the provided choi matrix is completely positive and trace preserving. This
    uses a series of repeat projection steps until a convergence is reached.

    Args:

        choi (np.ndarray) : The matrix which is to be projected.

        max_iter (int) : The max number of iterations that can be performed by
            the projection algorithm.

    Returns:

        np.ndarray : The calculated choi matrix, which meets the required
            conditions to be physical.

    """
    x_0 = _vec(choi)
    dim = x_0.shape[0]
    # Initialize most quantities as a zero matrix
    p_0 = np.array([0] * dim)
    q_0 = np.array([0] * dim)
    y_0 = np.array([0] * dim)
    # Run for maximum number of iterations
    for _ in range(max_iter):
        # Calculate updated quantiles
        y_k = _vec(_tp_proj(_unvec(x_0 + p_0)))
        p_k = x_0 + p_0 - y_k
        x_k = _vec(_cp_proj(_unvec(y_k + q_0)))
        q_k = y_k + q_0 - x_k
        # Stopping condition (see paper)
        if (
            np.linalg.norm(p_0 - p_k) ** 2
            + np.linalg.norm(q_0 - q_k) ** 2
            + abs(2 * np.conj(p_0) @ (x_k - x_0))
            + abs(2 * np.conj(q_0) @ (y_k - y_0))
        ) < 10**-4:
            break
        # Update values
        y_0, p_0, x_0, q_0 = y_k, p_k, x_k, q_k

    return _unvec(x_k)


def _cp_proj(choi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Performs the CP part of the projection by enforcing that the eigenvalues
    of the matrix are all positive.
    """
    vals, vecs = _find_eigen_data(choi)
    d = np.diag([max(v, 0) for v in vals])
    return vecs @ d @ np.conj(vecs.T)


def _tp_proj(choi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Performs the TP part of the projection by enforcing that the partial
    trace of the choi matrix is equal to the identity matrix.
    """
    # Compute partial trace
    dim = int(choi.shape[0] ** 0.5)
    partial_trace = choi.reshape(np.tile([dim, dim], 2))
    partial_trace = np.einsum(partial_trace, np.array([0, 1, 2, 1])).astype(  # type: ignore[arg-type]
        complex
    )
    partial_trace = partial_trace.reshape(dim, dim)  # type: ignore[assignment]
    # Then find the variation and subtract this from the choi matrix
    variation = partial_trace - np.identity(dim)
    return choi - np.kron(variation / dim, np.identity(dim))


def _find_eigen_data(
    matrix: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Calculates and returns the eigenvalues and eigenvectors of a provided
    matrix. These are returned from the smallest to the largest eigenvalue.
    """
    return np.linalg.eigh(matrix)


def _vec(mat: NDArray[T]) -> NDArray[T]:
    """
    Applies flatten operation to a provided matrix to convert it into a vector.
    """
    return mat.flatten()


def _unvec(mat: NDArray[T]) -> NDArray[T]:
    """
    Takes a provided vector and converts it into a square matrix.
    """
    dim = int(mat.shape[0] ** 0.5)
    return mat.reshape(dim, dim)
