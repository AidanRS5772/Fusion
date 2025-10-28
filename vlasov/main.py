from math import sqrt
import numpy as np
from scipy.integrate import solve_bvp
import plotly.graph_objects as go


class Vlasov:
    # NOTE:
    # V MeV
    # r cm
    # R cm
    # n scaled by 10^-12
    # l dimensionless
    def __init__(self, V, r, R, lam, n) -> None:
        self.V = V
        self.r = r
        self.R = R
        self.lam = lam
        self.n = n

        self._Lr = n * r**2 * sqrt(V * lam) * 9.45319767
        self._LR = n * R**2 * sqrt(V * lam) * 9.45319767
        self._inner_sol = self.solve_innner()
        self._outer_sol = self.solve_outer()

    def solve_innner(self, eps=1e-6, init_nodes=int(1e3), tol=1e-6, max_nodes=int(1e4)):
        def ode(x, y):
            y0, y1 = y
            return np.vstack([y1, -self._Lr * x * np.exp(-y0 / x)])

        def bc(ya, yb):
            return np.array([eps * ya[1] - ya[0], yb[0]])

        def jac(x, y):
            y0, _ = y

            dy = np.array(
                [
                    [np.zeros_like(x), np.ones_like(x)],
                    [self._Lr * y0 * np.exp(-y0 / x), np.zeros_like(x)],
                ]
            )

            dx = np.array(
                [np.zeros_like(x), -self._Lr * np.exp(-y0 / x) * (1 + y0 / x)]
            )

            return dy, dx

        def bc_jac(ya, yb):
            dya = np.array([[1.0, 0.0], [0.0, 0.0]])
            dyb = np.array([[0.0, 0.0], [1.0, 0.0]])
            return dya, dyb

        x = np.linspace(eps, 1, init_nodes)
        init_y = np.zeros((2, x.size))

        return solve_bvp(
            ode, bc, x, init_y, fun_jac=jac, bc_jac=bc_jac, tol=tol, max_nodes=max_nodes
        )

    def solve_outer(self, init_nodes=int(1e3), tol=1e-6, max_nodes=int(1e4)):
        def ode(x, y):
            y0, y1 = y
            return np.vstack(
                [
                    y1,
                    -self._LR
                    * x
                    * np.exp(
                        -(x - self.r / self.R) / (x * self.lam * (1 - self.r / self.R))
                        - y0 / x
                    ),
                ]
            )

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        def jac(x, y):
            y0, _ = y

            dy = np.array(
                [
                    [np.zeros_like(x), np.ones_like(x)],
                    [
                        self._LR
                        * y0
                        * np.exp(
                            -(x - self.r / self.R)
                            / (x * self.lam * (1 - self.r / self.R))
                            - y0 / x
                        ),
                        np.zeros_like(x),
                    ],
                ]
            )

            dx = np.array(
                [
                    np.zeros_like(x),
                    -self._LR
                    * np.exp(
                        -(x - self.r / self.R) / (x * self.lam * (1 - self.r / self.R))
                        - y0 / x
                    )
                    * (1 + (y0 + self.r / (self.R - self.r)) / x),
                ]
            )

            return dy, dx

        def bc_jac(ya, yb):
            dya = np.array([[1.0, 0.0], [0.0, 0.0]])
            dyb = np.array([[0.0, 0.0], [1.0, 0.0]])
            return dya, dyb

        x = np.linspace(self.r / self.R, 1, init_nodes)
        init_y = np.zeros((2, x.size))

        return solve_bvp(
            ode, bc, x, init_y, fun_jac=jac, bc_jac=bc_jac, tol=tol, max_nodes=max_nodes
        )

    def U(self, rho):
        def inner(rho):
            x = rho / self.r
            return abs(self.V) * self.lam * self._inner_sol.sol(x)[0] / x

        def outer(rho):
            x = rho / self.R
            return abs(self.V) * self.lam * self._outer_sol.sol(x)[0] / x

        U_rho = np.piecewise(rho, [rho <= self.r, rho > self.r], [inner, outer])

        return U_rho if rho.ndim else U_rho.item

    def phase_density(self, rho, v_rho):
        def inner(rho):
            x = rho / self.r
            return np.exp(-self._inner_sol.sol(x)[0] / x)

        def outer(rho):
            x = rho / self.R
            return np.exp(
                (x - self.r / self.R) / (x * self.lam * (1 - self.r / self.R))
                - self._outer_sol.sol(x)[0] / x
            )

        rho = np.asarray(rho)
        v_rho = np.asarray(v_rho)

        if rho.size != v_rho.size:
            raise ValueError("rho and v_rho must be of the same size")

        f_rho = np.piecewise(rho, [rho <= self.r, rho > self.r], [inner, outer])
        f = self.n * f_rho * np.exp(-1.043450424 * v_rho**2 / (self.V * self.lam))

        return f if rho.ndim else f.item

    def number_density(self, rho):
        def inner(rho):
            x = rho / self.r
            return np.exp(-self._inner_sol.sol(x)[0] / x)

        def outer(rho):
            x = rho / self.R
            return np.exp(
                (x - self.r / self.R) / (x * self.lam * (1 - self.r / self.R))
                - self._outer_sol.sol(x)[0] / x
            )

        rho = np.asarray(rho)
        f_rho = np.piecewise(rho, [rho <= self.r, rho > self.r], [inner, outer])
        f = self.n * 0.4157260273 * (self.V * self.lam) ** (1.5) * f_rho

        return f if rho.ndim else f.item


def main():
    v = Vlasov(1, 2.5, 25, 1, 1)

    fig = go.Figure()

    X = np.linspace(0, 25, 1000)
    Y = v.U(X)

    fig.add_scatter(x=X, y=Y, mode="lines")

    fig.show()


if __name__ == "__main__":
    main()
