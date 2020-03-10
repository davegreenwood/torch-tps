import unittest
import torch

from thinplatespline.batch import (
    K_matrix as K_batch, P_matrix as P_batch, TPS, TPS_coeffs)
from thinplatespline.tps import K_matrix, P_matrix, tps_coefs


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestTPS(unittest.TestCase):

    def test_grid_size(self):
        n, h, w, k = 1, 50, 20, 2
        size = (h, w)
        tps = TPS(size=size, device=DEVICE)
        self.assertTrue(tps.device == DEVICE)
        self.assertTrue((n, h*w, k) == tps.grid.shape)

    def test_kmatrix(self):
        n, k, d = 3, 5, 2
        x = torch.randn(k, d)
        xb = torch.stack([x] * n)
        k_b = K_batch(xb, xb)
        k_m = K_matrix(x, x)
        self.assertTrue(k_b.shape == (n, k, k))
        self.assertTrue(k_m.shape == (k, k))
        self.assertTrue(torch.allclose(k_b[0], k_m, atol=1e-5))

    def test_pmatrix(self):
        n, k, d = 3, 5, 2
        x = torch.randn(k, d)
        xb = torch.stack([x] * n)
        k_b = P_batch(xb)
        k_m = P_matrix(x)
        self.assertTrue(k_b.shape == (n, k, 3))
        self.assertTrue(k_m.shape == (k, 3))
        self.assertTrue(torch.allclose(k_b[0], k_m, atol=1e-5))

    def test_tps_coeffs(self):
        n, k, d = 3, 5, 2
        coefb = TPS_coeffs()
        x = torch.randn(k, d)
        y = torch.randn(k, d)
        xb = torch.stack([x] * n)
        yb = torch.stack([y] * 1)
        wb, ab = coefb(xb, yb)
        w, a = tps_coefs(x, y)
        self.assertTrue(wb.shape == (n, k, d))
        self.assertTrue(ab.shape == (n, d + 1, d))
        self.assertTrue(w.shape == (k, d))
        self.assertTrue(a.shape == (d + 1, d))
        self.assertTrue(torch.allclose(ab[0], a, atol=1e-5))
        self.assertTrue(torch.allclose(wb[0], w, atol=1e-5))

    def test_tps_warp(self):
        n, h, w, k, d = 3, 15, 15, 5, 2
        warp = TPS(size=(h, w))
        x = torch.randn(k, d)
        y = torch.rand(k, d)
        xb = torch.stack([x] * n)
        yb = torch.stack([y] * 1)
        grid = warp(xb, yb)
        self.assertTrue(grid.shape == (n, h, w, d))
