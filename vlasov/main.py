import numpy as np
import scipy as sp
import plotly.graph_objects as go

mD = 2.08690083  # [MeV][cm/ns]^-2 mass of a deuteron (m = E/c^2)
ep = 1.8095128  # [MeV][cm] x10^-12
kB = 86.17333262  # [MeV][K]^-1 x10^-12
nTorr = 0.83219  # [MeV][cm]^-3


class Vlasov:
    def system_equations(self, eps):
        def in_sys(x, y):
            return np.vstack((y[1], -self._C * x * np.exp(-y[0] / (self.l * x))))

        def in_jac(x, y):
            m = x.size
            J = np.zeros((2, 2, m))
            J[0, 1, :] = 1.0
            J[1, 0, :] = (self._C / self.l) * np.exp(-y[0] / (self.l * x))

            return J

        def in_bc(ya, yb):
            return np.array([eps * ya[1] - ya[0], yb[0]])

        def in_bc_jac(ya, yb):
            return np.array([[-1.0, eps], [0.0, 0.0]]), np.array(
                [[0.0, 0.0], [0.0, 1.0]]
            )

        def out_sys(x, y):
            exp_val = (
                self.V * (self.R / x) * (x - self.r) / (self.R - self.r) - y[0] / x
            ) / self.l
            return np.vstack((y[1], -self._C * x * np.exp(exp_val)))

        def out_jac(x, y):
            m = x.size
            J = np.zeros((2, 2, m))
            J[0, 1, :] = 1.0
            exp_val = (
                self.V * (self.R / x) * (x - self.r) / (self.R - self.r) - y[0] / x
            ) / self.l
            J[1, 0, :] = (self._C / self.l) * np.exp(exp_val)
            return J

        def out_bc(ya, yb):
            return np.array([ya[0], yb[0]])

        def out_bc_jac(ya, yb):
            return np.array([[1.0, 0.0], [0.0, 0.0]]), np.array(
                [[0.0, 0.0], [1.0, 0.0]]
            )

        return in_sys, in_jac, in_bc, in_bc_jac, out_sys, out_jac, out_bc, out_bc_jac

    def init_solve(self, size=100, eps=1e-6, max_nodes=int(1e5), sol_tol=1e-4):
        in_x = np.linspace(eps, self.r, size // 2)
        out_x = np.linspace(self.r, self.R, size // 2)

        y_est = np.zeros((2, size // 2))

        in_sys, in_jac, in_bc, in_bc_jac, out_sys, out_jac, out_bc, out_bc_jac = (
            self.system_equations(eps)
        )

        in_sol = sp.integrate.solve_bvp(
            in_sys,
            in_bc,
            in_x,
            y_est,
            fun_jac=in_jac,
            bc_jac=in_bc_jac,
            max_nodes=max_nodes,
            tol=sol_tol,
        )
        out_sol = sp.integrate.solve_bvp(
            out_sys,
            out_bc,
            out_x,
            y_est,
            fun_jac=out_jac,
            bc_jac=out_bc_jac,
            max_nodes=max_nodes,
            tol=sol_tol,
        )

        if not in_sol.success:
            raise ValueError(f"Interior Solution Error: {in_sol.message}")

        if not out_sol.success:
            raise ValueError(f"Exterior Solution Error: {out_sol.message}")

        return in_sol, out_sol

    def __init__(self, R, r, V, n, l) -> None:
        self.R = R
        self.r = r
        self.V = V

        self.n = n
        self.l = l

        self._C = n * ep * ((2 * np.pi * l / mD) ** 1.5)
        self._in_sol, self._out_sol = self.init_solve()

        self.N = self.find_N()
        self.E = self.find_E()

    def recompute(self, n, l, eps=1e-6, max_nodes=int(1e9), sol_tol=1e-4):
        self.n = n
        self.l = l

        in_x = self._in_sol.x
        out_x = self._out_sol.x

        in_y_est = self._in_sol.y
        out_y_est = self._out_sol.y

        in_sys, in_jac, in_bc, in_bc_jac, out_sys, out_jac, out_bc, out_bc_jac = (
            self.system_equations(eps)
        )

        in_sol = sp.integrate.solve_bvp(
            in_sys,
            in_bc,
            in_x,
            in_y_est,
            fun_jac=in_jac,
            bc_jac=in_bc_jac,
            max_nodes=max_nodes,
            tol=sol_tol,
        )
        out_sol = sp.integrate.solve_bvp(
            out_sys,
            out_bc,
            out_x,
            out_y_est,
            fun_jac=out_jac,
            bc_jac=out_bc_jac,
            max_nodes=max_nodes,
            tol=sol_tol,
        )

        if not in_sol.success:
            raise ValueError(f"Interior Solution Error: {in_sol.message}")

        if not out_sol.success:
            raise ValueError(f"Exterior Solution Error: {out_sol.message}")

        self._in_sol = in_sol
        self._out_sol = out_sol

        self.N = self.find_N()
        self.E = self.find_E()

    # 10^12
    def find_N(self, eps=1e-6):
        return sp.integrate.quad(lambda x: (x**2) * self.density(x), eps, self.R)[0]

    # Units of MeV 10^12 or .160218 J
    def find_E(self, eps=1e-6):
        return (
            3 * self.l * self.N
            + (2 * np.pi / ep)
            * sp.integrate.quad(lambda x: (x * self.dU(x)) ** 2, eps, self.R)[0]
        )

    def U(self, x):
        return np.where(
            x < self.r,
            self.V + self._in_sol.sol(x)[0] / x,
            self.V * (self.r / x) * (self.R - x) / (self.R - self.r)
            + self._out_sol.sol(x)[0] / x,
        )

    def dU(self, x):
        return np.where(
            x < self.r,
            (self._in_sol.sol(x)[1] - self._in_sol.sol(x)[0] / x) / x,
            -(self.V / (1 / self.r - 1 / self.R)) / (x**2)
            + (self._out_sol.sol(x)[1] - self._out_sol.sol(x)[0] / x) / x,
        )

    def density(self, x):
        def F(x):
            return np.where(
                x < self.r, 0, (self.R / x) * (x - self.r) / (self.R - self.r)
            )

        return (
            2
            * np.pi
            * self.n
            * ((2 * np.pi * self.l / mD) ** 1.5)
            * np.exp((self.V * F(x) - self.U(x)) / self.l)
        )


# R [cm]
# r [cm]
# P [nTorr]
# V [MeV]
# T [K]
def find_solution(R, r, P, V, T, error_val=1e6):
    N = nTorr * P * 4 * np.pi * R**3 / (3 * kB * T)
    E = 1.5 * N * kB * T * 1e-12 + (2 * np.pi / ((1 / r - 1 / R) * ep)) * (V**2)
    print(f"Number of Particles: {N} x 10^12")
    print(f"Energy: {E} [MeV] x 10^12")

    n_min, n_max = 1e-8, 1
    l_min, l_max = 1e-2, 100

    init_n = 0.5
    init_l = 0.01

    Vlasov_Solution = Vlasov(R, r, V, init_n, init_l)

    def res(x):
        log_n, log_l = x
        try:
            Vlasov_Solution.recompute(np.exp(log_n), np.exp(log_l))
            return np.array(
                [np.log(Vlasov_Solution.N / N), np.log(Vlasov_Solution.E / E)]
            )
        except Exception:
            return np.array([error_val, error_val])

    ls = sp.optimize.least_squares(
        res,
        x0=[np.log(init_n), np.log(init_l)],
        bounds=([np.log(n_min), np.log(l_min)], [np.log(n_max), np.log(l_max)]),
        method="trf",
        loss="huber",  # good balance; linear near 0, L1 in tails
        f_scale=1.0,  # residuals are ~O(1) in log space when close
        max_nfev=3000,
        verbose=2,  # watch progress; remove when happy
    )
    if not ls.success:
        raise RuntimeError(ls.message)

    n_opt, l_opt = ls.x
    return Vlasov(R, r, V, np.exp(n_opt), np.exp(l_opt))


def main():
    R = 25
    r = 5
    sol = find_solution(R, r, 1, -0.1, 300)
    print(f"Found n: {sol.n}")
    print(f"Found l: {sol.l}")
    print(f"Found Number of Particles: {sol.N} x10^12")
    print(f"Found Energy: {sol.E} [MeV] x10^12")

    fig = go.Figure()
    x = np.linspace(1e-3, R, 1000)
    y = sol.density(x)
    fig.add_scatter(x=x, y=y, mode="lines")
    fig.show()


if __name__ == "__main__":
    main()
