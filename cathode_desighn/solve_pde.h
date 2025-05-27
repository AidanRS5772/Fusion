#ifndef SOLVE_PDE_H
#define SOLVE_PDE_H

#include <array>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_in.h>
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
#include <string>

using namespace dealii;

class SolvePDE {
  public:
    SolvePDE(const std::string &file_name_, const double voltage_, const double scale, const int order,
             int max_iter = 1000, double tol = 1e-6)
        : mesh_file_name(file_name_), voltage(voltage_), scale(scale), finite_element(order) {
        std::cout << "Starting PDE Solve..." << std::endl;
        read_mesh();
        setup_system();
        assemble();
        solve(max_iter, tol);
    }

    double V(const std::array<double, 3> &p) {
        Point<3> point(p[0] / scale, p[1] / scale, p[2] / scale);
        return VectorTools::point_value(dof_handler, solution, point);
    }

    std::array<double, 3> E(const std::array<double, 3> &p) {
        Point<3> point(p[0] / scale, p[1] / scale, p[2] / scale);
        Tensor<1, 3, double> val = VectorTools::point_gradient(dof_handler, solution, point);
        return {-val[0] / scale, -val[1] / scale, -val[2] / scale};
    }

  private:
    std::string mesh_file_name;
    double voltage;
    double scale;

    Triangulation<3> triangulation;
    FE_SimplexP<3> finite_element;
    DoFHandler<3> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;

    void read_mesh() {
        GridIn<3> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file("../mesh_cache/" + mesh_file_name);

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
        const MappingFE<3> mapping(finite_element);
        FEValues<3> fe_values(mapping, finite_element, quadrature_formula,
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
                        cell_matrix(i, j) += fe_values.shape_grad(i, q_point) *
                                             fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point);
                    }
                }
            }

            cell->get_dof_indices(local_dof_indices);
            for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                    system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
        }

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(1),
                                                 Functions::ConstantFunction<3>(-voltage), boundary_values);
        VectorTools::interpolate_boundary_values(dof_handler, types::boundary_id(2),
                                                 Functions::ZeroFunction<3>(), boundary_values);
        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

        std::cout << " assembly finished." << std::endl;
    }

    void solve(int max_iter, double tol) {
        SolverControl solver_control(max_iter, tol * system_rhs.l2_norm());
        SolverCG<Vector<double>> solver(solver_control);
        solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

        std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence."
                  << std::endl;
    }
};

#endif
