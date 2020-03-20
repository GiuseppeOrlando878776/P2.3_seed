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

#include <deal.II/base/thread_management.h>

using namespace dealii;



int main() {
  const int dim = 2;

  Triangulation<dim>   triangulation;
  GridGenerator::hyper_shell(triangualtion,Point<dim>(), 0.5, 1.0)
  triangulation.refine_global();

  Threads::ThreadGroup<void> thread_group;

  /*We fake my_direction as "FEValues"*/
  Point<dim> my_direction;
  my_direction[0] = 1.0;

  //Fake rhs vector
  double average_scalar_product = 0.0;

  double my_scalar_product = 0.0;

  Point<dim> PerTaskData;

  auto compute_on_one_cell[&cell,&cell_centers](const typename Triangulation<dim>::active_cell_iterator& cell,
                                                   Point<dim>& direction, double& my_scalar_product) {
     my_scalar_product = cell->center()*direction;

  };

  auto copy_from_one_cell = [&average_scalar_product](const double& my_scalar_product) {
     average_scalar_product += my_scalar_product;
  };


  for(auto& cell: triangulation.active_cell_iterators()) {
     compute_on_one_cell(cell,my_direction,my_scalar_product);
     copy_from_one_cell(my_scalar_product);
  }

  //Equivalently we can use WorkStream
  WorkStream::run(triangulation.begin_active(), triangulation.end_active(), compute_on_one_cell, copy_from_one_cell,
                  my_direction, my_scalar_product);

  //std::vector<Point<dim>> cell_centers(triangulation.n_active_cells());

  //auto sub_ranges = Threads::split_range(triangulation.begin_active(),
  /*                                       triangulation.end_active(), 8)
  for(const auto& cell_range:sub_ranges) {
    auto my_function = [&cell] {
        std::cout<<"Cell: "<<cell<<", "<<cell->center()<<std::endl();
    }


  for(const auto& cell: triangulation.active_cell_iterators()) {
    auto my_function = [&cell] {
        std::cout<<"Cell: "<<cell<<", "<<cell->center()<<std::endl();
    }

    //my_function();
    thread_group += Threads::new_thread(my_function);
  }
  thread_group.join_all();
  */
}

