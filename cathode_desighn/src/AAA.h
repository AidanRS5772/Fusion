#ifndef AAA_H
#define AAA_H

#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>

using VectorXd = Eigen::VectorXd;

class AAA {
  public:
    AAA(const std::vector<double> &Z_vec,
        const std::vector<double> &F_vec,
        const double abs_tol = 1e-4,
        const double rel_tol = 1e-3)
        : M(F_vec.size() - 2), F(M), Z(M), f(2), z(2), w(2), l(2), z_diff(2) {
        f[0] = F_vec.front();
        z[0] = Z_vec.front();
        f[1] = F_vec.back();
        z[1] = Z_vec.back();
        for (size_t i = 0; i < M; i++) {
            F[i] = F_vec[i + 1];
            Z[i] = Z_vec[i + 1];
        }
        update_wieghts();

        while (next_support(abs_tol, rel_tol)) {
            update_wieghts();
        }

        std::cout << "Aproximation Found with m = " << m << std::endl;
    }

    double eval(const double r) const {
        z_diff = r - z.array();
        Eigen::Index min_idx;
        double min_abs = z_diff.cwiseAbs().minCoeff(&min_idx);
        if (min_abs < 1e-15) {
            return f[min_idx];
        }
        l = z_diff.prod() / z_diff.array();
        return l.dot(wf) / l.dot(w);
    }

  private:
    size_t M, m = 2;
    VectorXd F, Z, f, z, w, wf;
    mutable VectorXd l, z_diff;

    void update_wieghts() {
        Eigen::MatrixXd C(M, m);
        for (size_t j = 0; j < m; j++) {
            const double z_j = z[j];
            for (size_t i = 0; i < M; i++) {
                C(i, j) = 1 / (Z[i] - z_j);
            }
        }
        const Eigen::MatrixXd A = F.asDiagonal() * C - C * f.asDiagonal();
        Eigen::JacobiSVD<Eigen::MatrixXd> SVD(A, Eigen::ComputeThinV);
        w = SVD.matrixV().col(SVD.matrixV().cols() - 1);
        wf = w.cwiseProduct(f);
    }

    bool next_support(const double abs_tol, const double rel_tol) {
        double max_abs_err = 0;
        double max_rel_err = 0;
        size_t max_abs_idx;
        size_t max_rel_idx;
        for (size_t i = 0; i < M; i++) {
            double abs_err = std::abs(F[i] - eval(Z[i]));
            double rel_err = std::abs(F[i]) > 1e-15 ? abs_err / std::abs(F[i]) : abs_err;
            if (max_abs_err < abs_err) {
                max_abs_err = abs_err;
                max_abs_idx = i;
            }
            if (max_rel_err < rel_err) {
                max_rel_err = rel_err;
                max_rel_idx = i;
            }
        }

        size_t idx;
        if (max_abs_err < abs_tol && max_rel_err > rel_tol) {
            idx = max_rel_idx;
        } else if (max_abs_err > abs_tol && max_rel_err < rel_tol) {
            idx = max_abs_idx;
        } else if (max_abs_err > abs_tol && max_rel_err > rel_tol) {
            if ((max_rel_err - rel_tol) > (max_abs_err / abs_tol - 1)) {
                idx = max_rel_idx;
            } else {
                idx = max_abs_idx;
            }
        } else {
            return false;
        }

        m++;
        M--;

        f.conservativeResize(m);
        f[m - 1] = F[idx];
        F.segment(idx, M - idx) = F.segment(idx + 1, M - idx);
        F.conservativeResize(M);

        z.conservativeResize(m);
        z[m - 1] = Z[idx];
        Z.segment(idx, M - idx) = Z.segment(idx + 1, M - idx);
        Z.conservativeResize(M);

        w.resize(m);
        wf.resize(m);
        l.resize(m);
        z_diff.resize(m);

        return true;
    }
};

#endif
