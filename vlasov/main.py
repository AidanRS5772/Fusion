from math import sqrt
import numpy as np

class Vlasov:
    # V MeV
    # r cm
    # R cm
    # n scaled by 10^-12
    # l dimensionless
    def __init__(self, V, r, R, l, n) -> None:
        self.V = V
        self.r = r
        self.R = R
        self.l = l
        self.n = n

        self._Lr = n*r**2*sqrt(V*l)*9.45315434
        self._LR = n*r**2*sqrt(V*l)*9.45315434

    def innner_sol(self):
        def ode(x, y):
            return np.vstack([y[1], -self._Lr*x*np.exp(-y[0]/x)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])




        

def main():



if __name__ == "__main__":
    main()
