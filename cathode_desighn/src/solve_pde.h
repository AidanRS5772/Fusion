#ifndef SOLVE_PDE_H
#define SOLVE_PDE_H

#include "octo_tree.h"
#include <Eigen/Dense>
#include <cstddef>
#include <cstdlib>
#include <deal.II/base/array_view.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_update_flags.h>
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
#include <deal.II/matrix_free/evaluation_flags.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_point_gradient.h>
#include <deal.II/numerics/vector_tools_point_value.h>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <ostream>
#include <string>

using namespace dealii;
using Vector3d = Eigen::Vector3d;

class SolvePDE {
  public:
    SolvePDE(
        const std::string &file_name_,
        const double voltage_,
        const double cathode_radius_,
        const size_t fem_order = 3,
        const size_t solve_max_iter = 10000,
        const double solve_tol = 1e-15)
        : mesh_file_name(file_name_), voltage(voltage_), cathode_radius(cathode_radius_), finite_element(fem_order), mapping(finite_element),
          grid_cache(triangulation), point_evaluator(mapping, finite_element, update_values | update_gradients),
          cached_cell_sol(finite_element.n_dofs_per_cell()) {
        MultithreadInfo::set_thread_limit(1);
        std::cout << "Starting PDE Solve..." << std::endl;
        read_mesh();
        setup_system();
        assemble();
        solve(solve_max_iter, solve_tol);
    }

    std::optional<double> V(const Vector3d &p) const {
        Point<3> point(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        auto cell_ref_opt = find_cell(point);
        if (!cell_ref_opt.has_value())
            return std::nullopt;
        auto [cell, ref_point] = cell_ref_opt.value();

        const auto dof_cell = cell->as_dof_handler_iterator(dof_handler);
        get_cached_dof_values(dof_cell);
        point_evaluator.reinit(dof_cell, ArrayView<const Point<3>>(&ref_point, 1));
        point_evaluator.evaluate(cached_cell_sol, EvaluationFlags::values);

        return point_evaluator.get_value(0);
    }

    std::optional<Vector3d> E(const Vector3d &p) const {
        Point<3> point(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        auto cell_ref_opt = find_cell(point);
        if (!cell_ref_opt.has_value())
            return std::nullopt;
        const auto [cell, ref_point] = cell_ref_opt.value();

        const auto dof_cell = cell->as_dof_handler_iterator(dof_handler);
        get_cached_dof_values(dof_cell);
        point_evaluator.reinit(dof_cell, ArrayView<const Point<3>>(&ref_point, 1));
        point_evaluator.evaluate(cached_cell_sol, EvaluationFlags::gradients);

        const auto g = point_evaluator.get_gradient(0);
        return Vector3d(-g[0] / cathode_radius, -g[1] / cathode_radius, -g[2] / cathode_radius);
    }

    std::optional<std::pair<double, Vector3d>> VE(const Vector3d &p) const {
        Point<3> point(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        auto cell_ref_opt = find_cell(point);
        if (!cell_ref_opt.has_value())
            return std::nullopt;
        const auto [cell, ref_point] = cell_ref_opt.value();

        const auto dof_cell = cell->as_dof_handler_iterator(dof_handler);
        get_cached_dof_values(dof_cell);
        point_evaluator.reinit(dof_cell, ArrayView<const Point<3>>(&ref_point, 1));
        point_evaluator.evaluate(cached_cell_sol, EvaluationFlags::gradients | EvaluationFlags::values);

        const auto g = point_evaluator.get_gradient(0);
        return std::pair{point_evaluator.get_value(0), Vector3d(-g[0] / cathode_radius, -g[1] / cathode_radius, -g[2] / cathode_radius)};
    }

    bool init_cache(const Vector3d &p) const {
        Point<3> point(p.x() / cathode_radius, p.y() / cathode_radius, p.z() / cathode_radius);
        auto [cell, ref_point] = GridTools::find_active_cell_around_point(grid_cache, point);
        if (cell == triangulation.end()) {
            return false;
        }
        last_cell = cell;
        cached_dof_cell = cell->as_dof_handler_iterator(dof_handler);
        cached_dof_cell->get_dof_values(solution, cached_cell_sol.begin(), cached_cell_sol.end());
        return true;
    }

  private:
    const std::string mesh_file_name;
    const double voltage;
    const double cathode_radius;

    const FE_SimplexP<3> finite_element;
    const MappingFE<3> mapping;
    Triangulation<3> triangulation;
    DoFHandler<3> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    GridTools::Cache<3, 3> grid_cache;
    mutable typename Triangulation<3>::active_cell_iterator last_cell;
    mutable FEPointEvaluation<1, 3> point_evaluator;
    mutable typename DoFHandler<3>::active_cell_iterator cached_dof_cell;
    mutable std::vector<double> cached_cell_sol;

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

        grid_cache.mark_for_update();
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
        FEValues<3> fe_values(mapping, finite_element, quadrature_formula, update_values | update_gradients | update_JxW_values);

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
                        cell_matrix(i, j) += fe_values.shape_grad(i, q_point) * fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
                    }
                }
            }

            cell->get_dof_indices(local_dof_indices);
            for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                    system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
        }

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(1), Functions::ConstantFunction<3>(-voltage), boundary_values);
        VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(2), Functions::ZeroFunction<3>(), boundary_values);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

        std::cout << " assembly finished." << std::endl;
    }

    void solve(int max_iter, double tol) {
        SolverControl solver_control(max_iter, tol * system_rhs.l2_norm());
        SolverCG<Vector<double>> solver(solver_control);
        solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

        std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;
    }

    std::optional<std::pair<typename Triangulation<3>::active_cell_iterator, Point<3>>> find_cell(const Point<3> &p) const {
        auto result = GridTools::find_active_cell_around_point(grid_cache, p, last_cell);
        if (result.first == triangulation.end()) {
            return std::nullopt;
        }
        last_cell = result.first;
        return result;
    }

    void get_cached_dof_values(const typename DoFHandler<3>::active_cell_iterator &dof_cell) const {
        if (dof_cell != cached_dof_cell) {
            cached_dof_cell = dof_cell;
            dof_cell->get_dof_values(solution, cached_cell_sol.begin(), cached_cell_sol.end());
        }
    }
};

#endif
