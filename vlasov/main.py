from math import sqrt
import numpy as np
from scipy.integrate import solve_bvp, simpson, quad
import plotly.graph_objects as go
from fixed_bvp import Fixed2ndOrderBVP
import time


class Vlasov:
    # V [MeV]
    # R [cm]
    # r [cm]
    # n [10^9]
    # b [MeV]^-1

    def __init__(self, V, R, r, n, b, init_nodes=int(1e3)):
        self.V = V
        self.R = R
        self.r = r
        self.n = n
        self.b = b

        self._Li = 0.09453197669 * n * r**2 * np.exp(b * V) / sqrt(b)
        self._lo = b * V * r / (R - r)
        self._Lo = 0.09453197669 * n * R**2 * np.exp(-self._lo) / sqrt(b)

        init_y = np.zeros((2, init_nodes))

        in_x = np.linspace(0, 1, init_nodes)
        self._in_sol = self._find_in_sol(in_x, init_y)

        self._in_sol_n = self._find_in_sol_n(np.zeros(len(self._in_sol.x)))
        self._in_sol_b = self._find_in_sol_b(np.zeros(len(self._in_sol.x)))

        out_x = np.linspace(r / R, 1, init_nodes)
        self._out_sol = self._find_out_sol(out_x, init_y)

        self._out_sol_n = self._find_out_sol_n(np.zeros(len(self._out_sol.x)))
        self._out_sol_b = self._find_out_sol_b(np.zeros(len(self._out_sol.x)))

        # N [10^12]
        self.N = self._find_N()
        self.N_n = self._find_N_n()
        self.N_b = self._find_N_b()

        # J
        self.E = self._find_E()
        self.E_n = self._find_E_n()
        self.E_b = self._find_E_b()

    def _find_N(self):
        return -6.94461548 * self.R * self._out_sol.y[1][-1] / self.b

    def _find_N_n(self):
        return -6.94461548 * self.R * self._out_sol_n.yp[-1] / self.b

    def _find_N_b(self):
        return -self.N / self.b - 6.94461548 * self.R * self._out_sol_b.yp[-1] / self.b

    def _find_E(self):
        in_x = self._in_sol.x[1:]
        in_y = self._in_sol.y[0][1:]
        in_yp = self._in_sol.y[1][1:]
        E = simpson((in_yp - in_y / in_x) ** 2, in_x)

        out_x = self._out_sol.x
        out_y = self._out_sol.y[0]
        out_yp = self._out_sol.y[1]
        E += simpson((out_yp - out_y / out_x) ** 2, out_x)

        return (
            0.16021766339999
            * (3.47230774 * E + 10.4169232 * self.R * out_yp[-1])
            / self.b**2
        )

    def _find_E_n(self):
        in_x = self._in_sol.x[1:]
        in_y = self._in_sol.y[0][1:]
        in_yp = self._in_sol.y[1][1:]
        in_y_n = self._in_sol_n.y[1:]
        in_yp_n = self._in_sol_n.yp[1:]
        E = simpson((in_yp - in_y / in_x) * (in_yp_n - in_y_n / in_x), in_x)

        out_x = self._out_sol.x
        out_y = self._out_sol.y[0]
        out_yp = self._out_sol.y[1]
        out_y_n = self._out_sol_n.y
        out_yp_n = self._out_sol_n.yp
        E += simpson((out_yp - out_y / out_x) * (out_yp_n - out_y_n / out_x), out_x)

        return (
            0.16021766339999
            * (6.94461548 * E + 10.4169232 * self.R * out_yp_n[-1])
            / self.b**2
        )

    def _find_E_b(self):
        in_x = self._in_sol.x[1:]
        in_y = self._in_sol.y[0][1:]
        in_yp = self._in_sol.y[1][1:]
        in_y_b = self._in_sol_b.y[1:]
        in_yp_b = self._in_sol_b.yp[1:]
        E = simpson((in_yp - in_y / in_x) * (in_yp_b - in_y_b / in_x), in_x)

        out_x = self._out_sol.x
        out_y = self._out_sol.y[0]
        out_yp = self._out_sol.y[1]
        out_y_b = self._out_sol_b.y
        out_yp_b = self._out_sol_b.yp
        E += simpson((out_yp - out_y / out_x) * (out_yp_b - out_y_b / out_x), out_x)

        return (
            0.16021766339999
            * (
                6.94461548 * E
                + 10.4169232 * self.R * (out_yp_b[-1] - out_yp[-1] / self.b)
            )
            / self.b**2
        )

    def recompute(self, n, b):
        dn = self.n - n
        db = self.b - b
        self.n = n
        self.b = b
        self._Li = 0.09453197669 * n * self.r**2 * np.exp(b * self.V) / sqrt(b)
        self._lo = b * self.V * self.r / (self.R - self.r)
        self._Lo = 0.09453197669 * n * self.R**2 * np.exp(-self._lo) / sqrt(b)

        in_x = self._in_sol.x

        in_dy_dn = np.vstack([self._in_sol_n.y, self._in_sol_n.yp])
        in_dy_db = np.vstack([self._in_sol_b.y, self._in_sol_b.yp])
        in_y = self._in_sol.y + in_dy_dn * dn + in_dy_db * db

        self._in_sol = self._find_in_sol(in_x, in_y)

        self._in_sol_n = self._find_in_sol_n(np.zeros(len(self._in_sol.x)))
        self._in_sol_b = self._find_in_sol_b(np.zeros(len(self._in_sol.x)))

        out_x = self._out_sol.x

        out_dy_dn = np.vstack([self._out_sol_n.y, self._out_sol_n.yp])
        out_dy_db = np.vstack([self._out_sol_b.y, self._out_sol_b.yp])
        out_y = self._out_sol.y + out_dy_dn * dn + out_dy_db * db

        self._out_sol = self._find_out_sol(out_x, out_y)

        self._out_sol_n = self._find_out_sol_n(np.zeros(len(self._out_sol.x)))
        self._out_sol_b = self._find_out_sol_b(np.zeros(len(self._out_sol.x)))

        self.N = self._find_N()
        self.N_n = self._find_N_n()
        self.N_b = self._find_N_b()

        self.E = self._find_E()
        self.E_n = self._find_E_n()
        self.E_b = self._find_E_b()

    def _find_in_sol(self, init_x, init_y, eps=1e-6, max_nodes=int(1e9), tol=1e-3):
        def F(x, y):
            return np.where(x < eps, 0, -self._Li * x * np.exp(-y / x))

        def F_y(x, y):
            return np.where(x < eps, 0, self._Li * np.exp(-y / x))

        def ode(x, y):
            y0, y1 = y
            return np.vstack([y1, F(x, y0)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        def jac(x, y):
            y0, _ = y

            return np.array(
                [
                    [np.zeros_like(x), np.ones_like(x)],
                    [F_y(x, y0), np.zeros_like(x)],
                ]
            )

        def bc_jac(_, __):
            db_dya = np.array([[1.0, 0], [0.0, 0.0]])
            db_dyb = np.array([[0.0, 0.0], [1.0, 0.0]])

            return db_dya, db_dyb

        with np.errstate(divide="ignore", invalid="ignore"):
            sol = solve_bvp(
                ode,
                bc,
                init_x,
                init_y,
                fun_jac=jac,
                bc_jac=bc_jac,
                tol=tol,
                max_nodes=max_nodes,
            )
            assert sol.success, "Inner solution did not converge"
            return sol

    def _find_out_sol(self, init_x, init_y, max_nodes=int(1e6), tol=1e-6):
        def ode(x, y):
            y0, y1 = y
            return np.vstack([y1, -self._Lo * np.exp((self._lo - y0) / x)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        def jac(x, y):
            y0, _ = y

            return np.array(
                [
                    [np.zeros_like(x), np.ones_like(x)],
                    [self._Lo * np.exp((self._lo - y0) / x), np.zeros_like(x)],
                ]
            )

        def bc_jac(_, __):
            db_dya = np.array([[1.0, 0.0], [0.0, 0.0]])
            db_dyb = np.array([[0.0, 0.0], [1.0, 0.0]])

            return db_dya, db_dyb

        with np.errstate(divide="ignore", invalid="ignore"):
            sol = solve_bvp(
                ode,
                bc,
                init_x,
                init_y,
                fun_jac=jac,
                bc_jac=bc_jac,
                tol=tol,
                max_nodes=max_nodes,
            )
            assert sol.success, "Outer solution did not converge"
            return sol

    def _find_in_sol_n(self, init_y, eps=1e-3):
        def F(x, y):
            return np.where(
                x < eps,
                0.0,
                self._Li * np.exp(-self._in_sol.y[0][1:-1] / x) * (y - x / self.n),
            )

        def Fy(x, _):
            return np.where(
                x < eps, 0.0, self._Li * np.exp(-self._in_sol.y[0][1:-1] / x)
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            sol = Fixed2ndOrderBVP(self._in_sol.x, init_y, F, Fy, (0, 0), 0)
            assert sol.converged, "Inner solution of partial n did not converge"
            return sol

    def _find_in_sol_b(self, init_y, eps=1e-3):
        def F(x, y):
            return np.where(
                x < eps,
                0.0,
                self._Li
                * np.exp(-self._in_sol.y[0][1:-1] / x)
                * (y + (1 / (2 * self.b) - self.V) * x),
            )

        def Fy(x, _):
            return np.where(
                x < eps, 0.0, self._Li * np.exp(-self._in_sol.y[0][1:-1] / x)
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            sol = Fixed2ndOrderBVP(self._in_sol.x, init_y, F, Fy, (0, 0), 0)
            assert sol.converged, "Inner solution of partial b did not converge"
            return sol

    def _find_out_sol_n(self, init_y):
        def F(x, y):
            return (
                self._Lo
                * np.exp((self._lo - self._out_sol.y[0][1:-1]) / x)
                * (y - x / self.n)
            )

        def Fy(x, _):
            return self._Lo * np.exp((self._lo - self._out_sol.y[0][1:-1]) / x)

        with np.errstate(divide="ignore", invalid="ignore"):
            sol = Fixed2ndOrderBVP(self._out_sol.x, init_y, F, Fy, (0, 0))
            assert sol.converged, "Outer solution of partial n did not converge"
            return sol

    def _find_out_sol_b(self, init_y):
        def F(x, y):
            return (
                self._Lo
                * np.exp((self._lo - self._out_sol.y[0][1:-1]) / x)
                * (y + ((0.5 - self._lo) * x + self._lo) / self.b)
            )

        def Fy(x, _):
            return self._Lo * np.exp((self._lo - self._out_sol.y[0][1:-1]) / x)

        with np.errstate(divide="ignore", invalid="ignore"):
            sol = Fixed2ndOrderBVP(self._out_sol.x, init_y, F, Fy, (0, 0))
            assert sol.converged, "Outer solution of partial b did not converge"
            return sol

    def U(self, rho, eps=1e-3):
        def inner(rho):
            xi = np.asarray(rho) / self.r
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(
                    xi < eps,
                    self._in_sol.sol(eps)[0] / (self.b * eps),
                    self._in_sol.sol(xi)[0] / (self.b * xi),
                )

        def outer(rho):
            xi = np.asarray(rho) / self.R
            return self._out_sol.sol(xi)[0] / (self.b * xi)

        rho_arr = np.asarray(rho)

        assert np.any(rho_arr < self.R) and np.any(rho_arr > 0), (
            "Evaluated outside valid region"
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(rho_arr < self.r, inner(rho), outer(rho))

        return res.item() if np.isscalar(rho) else res

    def Up(self, rho, eps=1e-3):
        def inner(rho):
            xi = np.asarray(rho) / self.r
            return np.where(
                xi < eps,
                0.0,
                (self._in_sol.sol(xi)[1] - self._in_sol.sol(xi)[0] / xi)
                / (self.r * self.b * xi),
            )

        def outer(rho):
            xi = np.asarray(rho) / self.R
            return (self._in_sol.sol(xi)[1] - self._in_sol.sol(xi)[0] / xi) / (
                self.R * self.b * xi
            )

        rho_arr = np.asarray(rho)

        assert np.any(rho_arr < self.R) and np.any(rho_arr > 0), (
            "Evaluated outside valid region"
        )

        res = np.where(rho_arr < self.r, inner(rho), outer(rho))

        return res.item() if np.isscalar(rho) else res

    def U_c(self, rho):
        rho_arr = np.asarray(rho)
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(
                rho_arr <= self.r,
                -self.V,
                -self.V * (1 / rho_arr - 1 / self.R) / (1 / self.r - 1 / self.R),
            )
        return res.item() if np.isscalar(rho) else res

    def density(self, rho):
        return (
            5.224167332
            * (self.n / (self.b ** (1.5)))
            * np.exp(-self.b * (self.U_c(rho) + self.U(rho)))
        )

    def velocity_dist(self, w, l, eps=1e-6):
        W = np.exp(-1.043450424 * self.b * w**2)
        L = (
            self.N
            * (self.b * 0.3321405858) ** (3 / 2)
            * np.exp(-1.043450424 * self.b * l / self.R**2)
        )
        L += (
            2.77416917
            * self.b ** (5 / 2)
            * l
            * quad(
                lambda r: self.Up(r) * np.exp(-1.043450424 * self.b * l / r**2) / r,
                eps,
                self.R,
            )[0]
        )
        return W * L


def plot_velocity_dist_3d(v, w_range=(-2, 2), l_range=(0, 100), n_points=50):
    # Create grid (fewer points for 3D to keep it responsive)
    w = np.linspace(w_range[0], w_range[1], n_points)
    l = np.linspace(l_range[0], l_range[1], n_points)
    W, L = np.meshgrid(w, l)

    # Evaluate function on grid
    Z = np.zeros_like(W)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = v.velocity_dist(W[i, j], L[i, j])

    # Create 3D surface
    fig = go.Figure(
        data=[
            go.Surface(
                x=W,
                y=L,
                z=Z,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Velocity Distribution"),
            )
        ]
    )

    fig.update_layout(
        title="Velocity Distribution Function (3D)",
        scene=dict(
            xaxis_title="w",
            yaxis_title="l",
            zaxis_title="f(w, l)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=900,
        height=700,
    )

    fig.show()


def main():
    V, R, r = 1, 25, 5
    n, b = 1, 1
    v = Vlasov(V, R, r, n, b)

    plot_velocity_dist_3d(v)


if __name__ == "__main__":
    main()
