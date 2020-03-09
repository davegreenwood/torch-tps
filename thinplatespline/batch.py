"""Thin plate splines for batches."""

import torch

DEVICE = torch.device("cpu")


def K_matrix(X, Y):
    """Calculates the upper-right (k, k) submatrix of the
        (k + 3, k + 3) shaped L matrix.

    Parameters
    ----------
    X : (N, k, 2) torch.tensor of k points in 2 dimensions.
    Y : (1, m, 2) torch.tensor of m points in 2 dimensions.

    Returns
    -------
    K : torch.tensor
    """

    device = X.device

    D = torch.sqrt(
        torch.pow(X[:, :, None, :] - Y[:, None, :, :], 2).sum(-1) + 1e-9)
    D2 = D * D
    D2[torch.isclose(D2, torch.zeros(1, device=device))] = 1.0
    K = D * D * torch.log(D2)
    return K


def P_matrix(X):
    """Makes the minor diagonal submatrix P
    of the (k + 3, k + 3) shaped L matrix.

    Stacks a column of 1s before the coordinate columns in X.

    Parameters
    ----------
    X : (N, k, 2) torch.tensor of k points in 2 dimensions.

    Returns
    -------
    P : (N, k, 3) tensor, which is 1 in the first column, and
        exactly X in the remaining columns.
    """
    n, k = X.shape[:2]
    device = X.device

    P = torch.ones(n, k, 3, device=device)
    P[:, :, 1:] = X
    return P


class TPS_coeffs(torch.nn.Module):
    """Finds the thin-plate spline coefficients for the tps
    function that interpolates from X to Y.

    Parameters
    ----------
    X : torch tensor (N, K, 2), eg. projected points.
    Y : torch tensor (1, K, 2), eg. a UV map.

    Returns
    -------
    W : torch.tensor. (N, K, 2), the non-affine part of the spline
    A : torch.tensor. (N, K+1, K) the affine part of the spline.
    """
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        """Override abstract function."""

        n, k = X.shape[:2]
        device = X.device

        Z = torch.zeros(1, k + 3, 2, device=device)
        P = torch.ones(n, k, 3, device=device)
        L = torch.zeros(n, k + 3, k + 3, device=device)
        K = K_matrix(X, X)

        P[:, :, 1:] = X
        Z[:, :k, :] = Y
        L[:, :k, :k] = K
        L[:, :k, k:] = P
        L[:, k:, :k] = P.permute(0, 2, 1)

        Q = torch.solve(Z, L)[0]
        # return W and A.
        return Q[:, :k], Q[:, k:]


class TPS(torch.nn.Module):
    """Calculate the thin-plate-spline (TPS) surface at xy locations.

    Thin plate splines (TPS) are a spline-based technique for data
    interpolation and smoothing.
    see: https://en.wikipedia.org/wiki/Thin_plate_spline

    Constructor Params:
    device: torch.device
    size: tuple Output grid size as HxW. Output image size, default (256. 256).

    Parameters
    ----------
    X : torch tensor (N, K, 2), eg. projected points.
    Y : torch tensor (1, K, 2), for example, a UV map.

    Returns
    -------
     grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
    """

    def __init__(self, size: tuple = (256, 256), device=DEVICE):
        super().__init__()
        H, W = size
        self.device = device
        self.tps = TPS_coeffs()
        self.grid = torch.ones(1, H, W, 2, device=device)
        self.grid[:, :, :, 0] = torch.linspace(-1, 1, W)
        self.grid[:, :, :, 1] = torch.linspace(-1, 1, H)[..., None]

    def forward(self, X, Y):
        """Override abstract function."""
        h, w = self.grid.shape[1:3]
        W, A = self.tps(X, Y)
        U = K_matrix(self.grid.view(-1, h*w, 2), X)
        P = P_matrix(self.grid.view(-1, h*w, 2))
        grid = P @ A + U @ W
        return grid.view(-1, h, w, 2)
