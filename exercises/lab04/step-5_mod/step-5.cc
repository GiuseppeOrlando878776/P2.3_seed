/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */



#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>


#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;



template <int dim>
class Step5 {
public:
  Step5();
  void run(const unsigned int n_cycles = 1, const unsigned int inital_ref = 3);

private:
  void make_grid(const unsigned int intial_ref = 3);
  void estimate_error();
  void mark_cell_for_refinement();
  void refine_grid();
  void refine_grid_locally();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results(const unsigned int cycle) const;
  void compute_error();

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  AffineConstraints<double> constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;

  Vector<float> error_estimator;

  Vector<double> L2_error_per_cell;
  Vector<double> H1_error_per_cell;

  std::vector<double> L2_errors;
  std::vector<double> H1_errors;


  /*manifactured solution*/
  FunctionParser<dim> exact_solution;

  /*manifactured rhs*/
  FunctionParser<dim> rhs_function;

  /*Utility to compute error tables*/
  ParsedConvergenceTable error_table;

  //mutable TimerOutput timer;
};



template <int dim>
double coefficient(const Point<dim> &p) {
  if (p.square() < 0.5*0.5)
    return 20;
  else
    return 1;
}



template <int dim>
Step5<dim>::Step5() : fe(1), dof_handler(triangulation),
                      exact_solution("exp(x)*exp(y)"),
                      rhs_function("-2*exp(x)*exp(y)"),
                      error_table({"u"},{{VectorTools::H1_norm,VectorTools::L2_norm}}) {}
                      //timer(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times) {}



template<int dim>
void Step5<dim>::make_grid(const unsigned int initial_ref) {
    //TimerOutput::Scope timer_section(timer, "Make Grid");

    const double left = -1;
    const double right = 1;
    GridGenerator::hyper_cube(triangulation,left,right);
    triangulation.refine_global(initial_ref);
    std::cout << "   Number of active cells: "
                << triangulation.n_active_cells()
                << std::endl
                << "   Total number of cells: "
                << triangulation.n_cells()
                << std::endl;
}


template<int dim>
void Step5<dim>::estimate_error() {
    error_estimator.reinit(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1),
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       solution, error_estimator);
}


template<int dim>
void Step5<dim>::mark_cell_for_refinement() {
   GridRefinement::refine_and_coarsen_fixed_number(triangulation, error_estimator, 0.33, 0.0);
   //GridRefinement::refine_and_coarsen_fixed_fraction(triangulation, error_estimator, 0.33, 0.0); /*This is for more 'uniform' errors*/
}


/*
template<int dim>
void Step5<dim>::refine_grid() {
    //TimerOutput::Scope timer_section(timer, "Refine Grid");

    triangulation.refine_global();

    std::cout << "   Number of active cells: "
                << triangulation.n_active_cells()
                << std::endl
                << "   Total number of cells: "
                << triangulation.n_cells()
                << std::endl;
}*/

/*
template<int dim>
void Step5<dim>::refine_grid() {
    for(const auto& cell: triangulation.active_cell_iterators()) {
        if(cell->center()[1] > 1e-10)
            cell->set_refine_flag();
    }
    triangulation.execute_coarsening_and_refinement();
}*/


template<int dim>
void Step5<dim>::refine_grid() {
    /*
    for(const auto& cell: triangulation.active_cell_iterators()) {
        if(cell->center()[1] + cell->center()[0] > 1e-10)
            cell->set_refine_flag();
    }*/
    triangulation.execute_coarsening_and_refinement();
}

/*This is a summary of estimate_error(), mark_cell_for_refinement() and final version of refine_grid() */
template<int dim>
void Step5<dim>::refine_grid_locally() {
    //TimerOutput::Scope timer_section(timer, "Refine Grid Locally");

    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler, QGauss<dim - 1>(fe.degree + 1),
                                       std::map<types::boundary_id, const Function<dim> *>(),
                                       solution, estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);

    triangulation.execute_coarsening_and_refinement();
}


template <int dim>
void Step5<dim>::setup_system() {
  //TimerOutput::Scope timer_section(timer,"Setup System");

  dof_handler.distribute_dofs(fe);

  constraints.clear();
  VectorTools::interpolate_boundary_values(dof_handler, 0, exact_solution, constraints);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp,constraints,false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}



template <int dim>
void Step5<dim>::assemble_system() {
  //TimerOutput::Scope timer_section(timer, "Assemble System");

  QGauss<dim>  quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs(dofs_per_cell);
  std::vector<double>  rhs_values(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for(const auto& cell: dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs = 0;

      const auto& q_points = fe_values.get_quadrature_points();
      rhs_function.value_list(q_points, rhs_values);

      for(unsigned int q_index=0; q_index < n_q_points; ++q_index) {
          const double current_coefficient = coefficient<dim>(fe_values.quadrature_point (q_index));
          for(unsigned int i=0; i < dofs_per_cell; ++i) {
              for(unsigned int j=0; j < dofs_per_cell; ++j)
                cell_matrix(i,j) += (current_coefficient*
                                     fe_values.shape_grad(i,q_index) *
                                     fe_values.shape_grad(j,q_index) *
                                     fe_values.JxW(q_index));

              cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                              rhs_values[q_index] *
                              fe_values.JxW(q_index));
          }
      }


      cell->get_dof_indices(local_dof_indices);
      /*
      for(unsigned int i=0; i<dofs_per_cell; ++i) {
          for(unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
      */
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }

  //std::map<types::global_dof_index,double> boundary_values;
  //VectorTools::interpolate_boundary_values(dof_handler, 0, exact_solution, constraints);
  //MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);*/
}



template <int dim>
void Step5<dim>::solve() {
  //TimerOutput::Scope timer_section(timer, "Solve");

  SolverControl           solver_control(1000, 1e-12, false, false);
  SolverCG<>              solver(solver_control);

  /*PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);*/

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  constraints.distribute(solution);

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}


template<int dim>
void Step5<dim>::compute_error() {
   //TimerOutput::Scope timer_section(timer,"Compute Error");

   L2_error_per_cell.reinit(triangulation.n_active_cells());
   H1_error_per_cell.reinit(triangulation.n_active_cells());

   QGauss<dim> error_quadrature(2*fe.degree + 1);
   VectorTools::integrate_difference(dof_handler,solution,exact_solution,L2_error_per_cell,
                                     error_quadrature,VectorTools::L2_norm);
   VectorTools::integrate_difference(dof_handler,solution,exact_solution,H1_error_per_cell,
                                     error_quadrature,VectorTools::H1_norm);

   L2_errors.push_back(L2_error_per_cell.l2_norm());
   H1_errors.push_back(H1_error_per_cell.l2_norm());
}


template <int dim>
void Step5<dim>::output_results(const unsigned int cycle) const {
  //TimerOutput::Scope timer_section(timer,"Output");

  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(L2_error_per_cell,"L2_error");
  data_out.add_data_vector(H1_error_per_cell,"H1_error");
  data_out.add_data_vector(error_estimator,"error_estimator");
  data_out.build_patches();

  std::ostringstream filename;
  filename<<"solution-"<<cycle<<"_local_ref_discontinuous_coeff.vtk";
  std::ofstream output(filename.str().c_str());

  data_out.write_vtk(output);
}


template <int dim>
void Step5<dim>::run(const unsigned int n_cycles, const unsigned int init_ref) {
  std::ofstream output("error_analysis_local_ref_discontinuos_coeff.dat");
  make_grid(init_ref);

  for(unsigned int cycle = 0; cycle <n_cycles; ++cycle) {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if(cycle != 0)
        refine_grid();

      setup_system();
      assemble_system();
      solve();
      compute_error();
      estimate_error();
      mark_cell_for_refinement();
      error_table.error_from_exact(dof_handler,solution,exact_solution);
      output_results(cycle);
  }
  error_table.output_table(std::cout);
  error_table.output_table(output);
}



int main() {
  Step5<2> laplace_problem_2d;
  unsigned int n_cycles;
  std::cout<<"Insert the number of cycles desired"<<std::endl;
  std::cin>>n_cycles;
  laplace_problem_2d.run(n_cycles);
  return 0;
}
