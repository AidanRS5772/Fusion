#ifndef SOLVE_PDE_H
#define SOLVE_PDE_H

#include "AAA.h"
#include "octo_tree.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <deal.II/base/array_view.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_point_gradient.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

using namespace dealii;
using Vector3d = Eigen::Vector3d;

class SolvePDE {
  public:
    SolvePDE(
        const std::string &file_name_,
        const double voltage_,
        const double cathode_radius_,
        const double anode_radius_,
        const double inner_radius,
        const double outer_radius,
        const size_t fem_order = 3,
        const size_t solve_max_iter = 1000,
        const double solve_tol = 1e-15,
        const size_t sphere_sample_size = 16,
        const size_t radius_sample_order = 5,
        const double aprox_abs_tol = 0.001,
        const double aprox_rel_tol = 0.01)
        : mesh_file_name(file_name_), voltage(voltage_), cathode_radius(cathode_radius_), anode_radius(anode_radius_),
          N(1 << radius_sample_order), n(sphere_sample_size), finite_element(fem_order), mapping(finite_element) {
        std::cout << "Starting PDE Solve..." << std::endl;
        read_mesh();
        setup_system();
        assemble();
        solve(solve_max_iter, solve_tol);

        const Eigen::VectorXd eigen_in_r = Eigen::VectorXd::LinSpaced(N, 0.1, inner_radius);
        const std::vector<double> in_r(eigen_in_r.data(), eigen_in_r.data() + eigen_in_r.size());
        const auto in_sample_points = make_sample_point(in_r);

        const Eigen::VectorXd eigen_out_r = Eigen::VectorXd::LinSpaced(N, outer_radius, anode_radius);
        const std::vector<double> out_r(eigen_out_r.data(), eigen_out_r.data() + eigen_out_r.size());
        const auto out_sample_points = make_sample_point(out_r);

        std::cout << "Sampling Feilds..." << std::flush;
        const auto &[V_in, E_in, V_out, E_out] = sample_feilds(in_sample_points, out_sample_points);
        std::cout << " finished." << std::endl;

        const std::vector<double> V_in_avg = V_avgs(in_r, V_in);
        const std::vector<double> V_out_avg = V_avgs(out_r, V_out);
        const std::vector<double> E_in_avg = E_avgs(in_sample_points, E_in);
        const std::vector<double> E_out_avg = E_avgs(out_sample_points, E_out);

        const auto V_in_vars = V_var(V_in_avg, V_in);
        const auto V_out_vars = V_var(V_out_avg, V_out);
        const auto E_in_vars = E_vars(in_sample_points, E_in_avg, E_in);
        const auto E_out_vars = E_vars(out_sample_points, E_out_avg, E_out);

        const std::optional<size_t> V_in_idx = find_inner_idx(V_in_vars, aprox_abs_tol, aprox_rel_tol);
        if (V_in_idx.has_value()) {
            V_ir = in_r[V_in_idx.value()];
        }
        const std::optional<size_t> V_out_idx = find_outer_idx(V_out_vars, aprox_abs_tol, aprox_rel_tol);
        if (V_out_idx.has_value()) {
            V_or = out_r[V_out_idx.value()];
        }
        const std::optional<size_t> E_in_idx = find_inner_idx(V_in_vars, aprox_abs_tol, aprox_rel_tol);
        if (V_in_idx.has_value()) {
            E_or = in_r[E_in_idx.value()];
        }
        const std::optional<size_t> E_out_idx = find_outer_idx(E_out_vars, aprox_abs_tol, aprox_rel_tol);
        if (E_out_idx.has_value()) {
            E_or = out_r[E_out_idx.value()];
        }
    }

    std::optional<double> V(const Vector3d &p) const {
        const double p_norm = p.norm();
        if (p_norm < V_ir.value_or(-1)) {
            return V_inner_approx->eval(p_norm);
        } else if (p_norm > V_or.value_or(std::numeric_limits<double>::max())) {
            return V_outer_approx->eval(p_norm);
        } else {
            return V_exact(p);
        }
    }

    std::optional<Vector3d> E(const Vector3d &p) const {
        const double p_norm = p.norm();
        if (p_norm < E_ir.value_or(-1)) {
            return E_inner_approx->eval(p_norm) * p.normalized();
        } else if (p_norm > E_or.value_or(std::numeric_limits<double>::max())) {
            return E_outer_approx->eval(p_norm) * p.normalized();
        } else {
            return E_exact(p);
        }
    }

  private:
    const std::string mesh_file_name;
    const double voltage;
    const double cathode_radius;
    const double anode_radius;
    const size_t N;
    const size_t n;
    std::optional<double> V_ir;
    std::optional<double> V_or;
    std::optional<double> E_ir;
    std::optional<double> E_or;
    std::optional<AAA> V_inner_approx;
    std::optional<AAA> V_outer_approx;
    std::optional<AAA> E_inner_approx;
    std::optional<AAA> E_outer_approx;

    const FE_SimplexP<3> finite_element;
    const MappingFE<3> mapping;
    Triangulation<3> triangulation;
    DoFHandler<3> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    void read_mesh() {
        GridIn<3> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file(std::string(PROJECT_ROOT) + "/mesh_cache/" + mesh_file_name);

        if (!input_file) {
            throw std::runtime_error("Cannot open mesh file: " + mesh_file_name);
        }

        try {
            grid_in.read_msh(input_file);
        } catch (const std::exception &e) {
            std::cerr << "Error reading mesh: " << e.what() << std::endl;
            throw;
        }
    }

    void setup_system() {
        dof_handler.reinit(triangulation);
        dof_handler.distribute_dofs(finite_element);
        DoFRenumbering::Cuthill_McKee(dof_handler);
        std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);
        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
        system_rhs = 0;
    }

    void assemble() {
        std::cout << "Starting system assembly..." << std::flush;

        const QGaussSimplex<3> quadrature_formula(finite_element.degree + 1);
        FEValues<3> fe_values(
            mapping,
            finite_element,
            quadrature_formula,
            update_values | update_gradients | update_JxW_values);

        const unsigned int dofs_per_cell = finite_element.n_dofs_per_cell();
        const unsigned int n_q_points = quadrature_formula.size();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            cell_matrix = 0;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        cell_matrix(i, j) += fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) *
                                             fe_values.JxW(q_point);
                    }
                }
            }

            cell->get_dof_indices(local_dof_indices);
            for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                    system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
        }

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(
            dof_handler,
            types::boundary_id(1),
            Functions::ConstantFunction<3>(-voltage),
            boundary_values);
        VectorTools::interpolate_boundary_values(
            dof_handler,
            types::boundary_id(2),
            Functions::ZeroFunction<3>(),
            boundary_values);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

        std::cout << " assembly finished." << std::endl;
    }

    void solve(int max_iter, double tol) {
        SolverControl solver_control(max_iter, tol * system_rhs.l2_norm());
        SolverCG<Vector<double>> solver(solver_control);
        solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

        std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    }

    std::optional<double> V_exact(const Vector3d &p) const {
        Point<3> point(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        try {
            return VectorTools::point_value(dof_handler, solution, point);
        } catch (...) {
            return std::nullopt;
        }
    }

    std::optional<Vector3d> E_exact(const Vector3d &p) const {
        Point<3> point(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        Tensor<1, 3, double> val;
        try {
            val = VectorTools::point_gradient(dof_handler, solution, point);
        } catch (...) {
            return std::nullopt;
        }

        Vector3d out(-val[0] / cathode_radius, -val[1] / cathode_radius, -val[2] / cathode_radius);
        return out;
    }

    std::vector<Vector3d> make_sample_point(const std::vector<double> radii) {
        const double phi = M_PI * (std::sqrt(5) - 1);
        std::vector<Vector3d> sample_points;
        sample_points.reserve(N * n);
        for (const double r : radii) {
            for (size_t i = 0; i < n; i++) {
                const double z = 1 - static_cast<double>(2 * i) / (n - 1);
                const double rad = std::sqrt(1 - z * z);
                sample_points.push_back(Vector3d(r * rad * std::cos(i * phi), r * rad * std::sin(i * phi), r * z));
            }
        }
        return sample_points;
    }

    std::tuple<
        std::vector<std::optional<double>>,
        std::vector<std::optional<Vector3d>>,
        std::vector<std::optional<double>>,
        std::vector<std::optional<Vector3d>>>
    sample_feilds(const std::vector<Vector3d> &in_sample_points, const std::vector<Vector3d> &out_sample_points) {
        std::vector<Point<3>> points;
        points.reserve(2 * N * n);
        for (const auto &p : in_sample_points) {
            points.emplace_back(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        }
        for (const auto &p : out_sample_points) {
            points.emplace_back(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        }

        Utilities::MPI::RemotePointEvaluation<3, 3> point_cache;
        point_cache.reinit(points, triangulation, mapping);

        auto pots = VectorTools::point_values<1>(point_cache, dof_handler, solution);
        auto grads = VectorTools::point_gradients<1>(point_cache, dof_handler, solution);

        std::vector<std::optional<double>> V_in(N * n);
        std::vector<std::optional<Vector3d>> E_in(N * n);
        for (size_t i = 0; i < N * n; i++) {
            if (pots[i] != 0) {
                V_in[i] = pots[i];
            }
            if (grads[i][0] != 0 && grads[i][1] != 0 && grads[i][2] != 0) {
                E_in[i] = Vector3d(
                    -grads[i][0] / cathode_radius,
                    -grads[i][1] / cathode_radius,
                    -grads[i][2] / cathode_radius);
            }
        }
        std::vector<std::optional<double>> V_out(N * n);
        std::vector<std::optional<Vector3d>> E_out(N * n);
        for (size_t i = 0; i < N * n; i++) {
            if (pots[i] != 0) {
                V_out[i] = pots[i + N * n];
            }
            if (grads[i + N * n][0] != 0 && grads[i + N * n][1] != 0 && grads[i + N * n][2] != 0) {
                E_out[i] = Vector3d(
                    -grads[i + N * n][0] / cathode_radius,
                    -grads[i + N * n][1] / cathode_radius,
                    -grads[i + N * n][2] / cathode_radius);
            }
        }

        return {V_in, E_in, V_out, E_out};
    }

    std::vector<double> V_avgs(const std::vector<double> &radii, const std::vector<std::optional<double>> &V_samples) {
        std::vector<double> avgs(N);
        for (size_t i = 0; i < N; i++) {
            double val_sum = 0;
            bool flag = true;
            for (size_t j = 0; j < n; j++) {
                const auto val = V_samples[n * i + j];
                if (val.has_value()) {
                    val_sum += val.value();
                } else {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                avgs[i] = val_sum / n;
            } else {
                const double r = radii[i];
                if (r < cathode_radius) {
                    avgs[i] = V_exact(Vector3d::Zero()).value();
                } else {
                    avgs[i] = (voltage / (1 / cathode_radius - 1 / anode_radius)) * (1 / r - 1 / anode_radius);
                }
            }
        }

        return avgs;
    }

    std::vector<double> E_avgs(
        const std::vector<Vector3d> &samples,
        const std::vector<std::optional<Vector3d>> &E_samples) {
        std::vector<double> avgs(N);
        for (size_t i = 0; i < N; i++) {
            double val_sum = 0;
            bool flag = true;
            for (size_t j = 0; j < n; j++) {
                const auto val = E_samples[n * i + j];
                const auto p = samples[n * i + j];
                if (val.has_value()) {
                    val_sum += val->dot(p.normalized());
                } else {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                avgs[i] = val_sum / n;
            } else {
                const double r = samples[n * i].norm();
                if (r < cathode_radius) {
                    avgs[i] = 0;
                } else {
                    avgs[i] = voltage / (r * r * (1 / cathode_radius - 1 / anode_radius));
                }
            }
        }
        return avgs;
    }

    std::vector<std::optional<std::pair<double, double>>> V_var(
        const std::vector<double> avgs,
        const std::vector<std::optional<double>> vals) {
        std::vector<std::optional<std::pair<double, double>>> vars(N);
        for (size_t i = 0; i < N; i++) {
            const double avg = avgs[i];
            double val_sqr_diff_sum = 0;
            bool flag = true;
            for (size_t j = 0; j < n; j++) {
                const std::optional<double> val_opt = vals[n * i + j];
                if (val_opt.has_value()) {
                    const double diff = val_opt.value() - avg;
                    val_sqr_diff_sum += diff * diff;
                } else {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                const double var = std::sqrt(val_sqr_diff_sum / (n - 1));
                vars[i] = {var, var / std::abs(avg)};
            }
        }
        return vars;
    }

    std::vector<std::optional<std::pair<double, double>>> E_vars(
        const std::vector<Vector3d> points,
        const std::vector<double> avgs,
        const std::vector<std::optional<Vector3d>> vals) {
        std::vector<std::optional<std::pair<double, double>>> vars(N);
        for (size_t i = 0; i < N; i++) {
            const double avg = avgs[i];
            double val_sqr_diff_sum = 0;
            bool flag = true;
            for (size_t j = 0; j < n; j++) {
                const Vector3d p = points[i * N + j];
                const std::optional<Vector3d> val_opt = vals[n * i + j];
                if (val_opt.has_value()) {
                    val_sqr_diff_sum += (val_opt.value() - avg * p).squaredNorm() / 3;
                } else {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                const double var = std::sqrt(val_sqr_diff_sum / (n - 1));
                vars[i] = {var, var / std::abs(avg)};
            }
        }
        return vars;
    }

    std::optional<size_t> find_inner_idx(
        const std::vector<std::optional<std::pair<double, double>>> &vars,
        const double abs_tol,
        const double rel_tol) {
        std::optional<size_t> idx;
        for (int i = N; i >= 0; i--) {
            if (vars[i].has_value()) {
                if (vars[i]->first < abs_tol || vars[i]->second < rel_tol) {
                    if (!idx.has_value()) {
                        idx = i;
                    }
                } else {
                    idx = std::nullopt;
                }
            }
        }
        return idx;
    }

    std::optional<size_t> find_outer_idx(
        const std::vector<std::optional<std::pair<double, double>>> &vars,
        const double abs_tol,
        const double rel_tol) {
        std::optional<size_t> idx;
        for (size_t i = 0; i < N; i++) {
            if (vars[i].has_value()) {
                if (vars[i]->first < abs_tol || vars[i]->second < rel_tol) {
                    if (!idx.has_value()) {
                        idx = i;
                    }
                } else {
                    idx = std::nullopt;
                }
            }
        }
        return idx;
    }
};

#endif
