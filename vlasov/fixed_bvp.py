import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_banded


class Fixed2ndOrderBVP:
    def __init__(
        self, x, init_y, F, Fy, boundary, yp0=None, tol=1e-4, max_iter=int(1e3)
    ):
        assert len(x) == len(init_y), (
            f"init_y must be the same length as x -> init_y : {len(init_y)} , x : {len(x)}"
        )

        self.N = len(x)
        self.tol = tol
        self.max_iter = max_iter
        self.converged = False
        self.iterations = None
        self.residual = None

        self.x = x[1:-1]
        self.y = init_y[1:-1]
        self.a, self.b = boundary
        self.F = F
        self.Fy = Fy

        self._hp = x[2:] - x[1:-1]
        self._hm = x[1:-1] - x[:-2]
        self._hd = x[2:] - x[:-2]

        # initialize residual vector
        self.R = np.empty_like(x[1:-1])
        self._find_R()

        # initialize fixed portion of Jacobian as banded matrix
        self._hi_Jf = np.array(
            [2 / (self._hp[i] * self._hd[i]) for i in range(self.N - 3)]
        )
        self._c_Jf = np.array(
            [2 / (self._hp[i] * self._hm[i]) for i in range(self.N - 2)]
        )
        self._lo_Jf = np.array(
            [2 / (self._hm[i] * self._hd[i]) for i in range(1, self.N - 2)]
        )
        self._Jf = np.vstack(
            [
                np.concatenate([[0], self._hi_Jf]),
                -self._c_Jf,
                np.concatenate([self._lo_Jf, [0]]),
            ]
        )

        # initialize Jacobian
        self._J = self._Jf.copy()
        self._find_J()

        # solving for y
        self.solve_for_y()

        # adding boundary conditions to y
        self.x = np.concatenate([[x[0]], self.x, [x[-1]]])
        self.y = np.concatenate([[self.a], self.y, [self.b]])

        assert len(self.x) == self.N and len(self.y) == self.N, (
            f"The final array of values does not have the same number of nodes as the initial -> {len(self.x)} != {self.N} and {len(self.y)} != {self.N}"
        )
        assert np.array_equal(self.x, x), (
            "The final nodes are not equal to the initial nodes"
        )

        self.spline_coefs = self._find_spline_coefs(yp0)

        last_b = (
            self.spline_coefs[-1, 1]
            + 2 * self.spline_coefs[-1, 2]
            + 3 * self.spline_coefs[-1, 3]
            + 4 * self.spline_coefs[-1, 4]
        )
        self.yp = np.concatenate([self.spline_coefs[:, 1], [last_b]])

    def _find_R(self):
        self.R[0] = (2 / self._hd[0]) * (
            (self.y[1] - self.y[0]) / self._hp[0] - (self.y[0] - self.a) / self._hm[0]
        )
        self.R[-1] = (2 / self._hd[-1]) * (
            (self.b - self.y[-1]) / self._hp[-1]
            - (self.y[-1] - self.y[-2]) / self._hm[-1]
        )

        i = np.arange(1, self.N - 3)
        self.R[1:-1] = (2 / self._hd[i]) * (
            (self.y[i + 1] - self.y[i]) / self._hp[i]
            - (self.y[i] - self.y[i - 1]) / self._hm[i]
        )

        self.R -= self.F(self.x, self.y)

    def _find_J(self):
        self._J = self._Jf.copy()
        self._J[1] -= self.Fy(self.x, self.y)

    def solve_for_y(self):
        for iter in range(self.max_iter):
            self._find_R()
            if norm(self.R, ord=np.inf) < self.tol:
                self.converged = True
                self.iterations = iter
                self.residual = norm(self.R, ord=2) / (self.N - 3)
                return
            self._find_J()

            self.y -= solve_banded((1, 1), self._J, self.R)

        self.converged = False
        self.iterations = self.max_iter
        self.residual = norm(self.R, ord=2) / (self.N - 3)

        print(
            f"Warning: Maximum iterations reached -> tol = {norm(self.R, ord=np.inf)}"
        )

    def _find_spline_coefs(self, yp0):
        y = self.y

        h = self.x[1:] - self.x[:-1]
        h_sq = h**2

        yppa = (2 / (h[0] + h[1])) * ((y[2] - y[1]) / h[1] - (y[1] - y[0]) / h[0])
        yppb = (2 / (h[-1] + h[-2])) * (
            (y[-1] - y[-2]) / h[-1] - (y[-2] - y[-3]) / h[-2]
        )
        ypp = np.concatenate([[yppa], self.F(self.x[1:-1], self.y[1:-1]), [yppb]])

        coefs = np.empty((self.N - 1, 5))

        def compute_coefs(i, b):
            coefs[i, 0] = y[i]
            coefs[i, 1] = b
            coefs[i, 2] = h_sq[i] * ypp[i] / 2
            coefs[i, 3] = (
                2 * (y[i + 1] - y[i])
                - h_sq[i] * ((5 / 6) * ypp[i] + (1 / 6) * ypp[i + 1])
                - 2 * b
            )
            coefs[i, 4] = (
                -(y[i + 1] - y[i])
                + h_sq[i] * ((1 / 3) * ypp[i] + (1 / 6) * ypp[i + 1])
                + b
            )

        if yp0:
            compute_coefs(0, yp0)
        else:
            b = y[1] - y[0] - (1 / 3) * h_sq[0] * ypp[0] - (1 / 6) * h_sq[0] * ypp[1]
            compute_coefs(0, b)

        for i in range(1, self.N - 1):
            b = (h[i] / h[i - 1]) * (
                coefs[i - 1, 1]
                + 2 * coefs[i - 1, 2]
                + 3 * coefs[i - 1, 3]
                + 4 * coefs[i - 1, 4]
            )
            compute_coefs(i, b)

        return coefs

    def sol(self, x):
        x_input = x
        x = np.asarray(x)
        x = np.atleast_1d(x)

        # Check if x values are within bounds
        if np.any(x < self.x[0]) or np.any(x > self.x[-1]):
            raise ValueError(
                f"Solution evaluated outside of range [{self.x[0]}, {self.x[-1]}]"
            )

        # Check if x is exactly at a node
        # Use np.isin with tolerance for floating point comparison
        at_node_mask = np.zeros(len(x), dtype=bool)
        node_indices = np.zeros(len(x), dtype=int)

        for i, x_val in enumerate(x):
            # Check if close to any node (within floating point tolerance)
            close_to_node = np.abs(self.x - x_val) < 1e-10
            if np.any(close_to_node):
                at_node_mask[i] = True
                node_indices[i] = np.argmin(np.abs(self.x - x_val))

        # For non-node points, find interval
        idx = np.searchsorted(self.x, x, side="right") - 1
        # Clip to valid interval range [0, N-2]
        idx = np.clip(idx, 0, len(self.x) - 2)
        idx = np.asarray(idx, dtype=np.intp)  # Ensure idx is treated as an array

        # Compute normalized position t for spline evaluation
        t = np.zeros_like(x)
        non_node = ~at_node_mask
        t[non_node] = (x[non_node] - self.x[idx[non_node]]) / (
            self.x[idx[non_node] + 1] - self.x[idx[non_node]]
        )

        # Evaluate spline for non-node points
        result = (
            self.spline_coefs[idx, 0]
            + self.spline_coefs[idx, 1] * t
            + self.spline_coefs[idx, 2] * t**2
            + self.spline_coefs[idx, 3] * t**3
            + self.spline_coefs[idx, 4] * t**4
        )

        # Override with exact node values where applicable
        result[at_node_mask] = self.y[node_indices[at_node_mask]]

        return result.item() if np.isscalar(x_input) else result
