/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2019 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 */


#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>

#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

template<int dim=2>
void print_info(const Triangulation<dim>& triangulation) {
    std::cout<<"Number of levels: "<<triangulation.n_levels()<<std::endl;
    std::cout<<"Number of cells: "<<triangulation.n_cells()<<std::endl;
    std::cout<<"Number of active cells: "<<triangulation.n_active_cells()<<std::endl;
}


void first_grid()
{
  Triangulation<2> triangulation;

  //GridGenerator::hyper_cube(triangulation);

  const Point<2> center(1, 0);
  const double   inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(
    triangulation, center, inner_radius, outer_radius, 0, true);

  triangulation.reset_all_manifolds();

  SphericalManifold<2> manifold(center);

  for(auto& cell: triangulation.active_cell_iterators()) {
    for(unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell;++f) {
        if(cell->face(f)->boundary_id()==0 && cell->center()[1]>1.0e-10)
        //if(cell->center()[1]>1.0e-10)
        //    cell->set_all_manifold_ids(100);
            cell->face(f)->set_manifold_id(100);
    }
  }

  const unsigned int ref_level = 4;

  triangulation.set_manifold(100,manifold);
  for(unsigned int i = 0; i < ref_level; ++i) {
    std::ofstream out("grid-1_reset_manifold_"+std::to_string(i)+".vtk");
    GridOut       grid_out;
    grid_out.write_vtk(triangulation, out);
    //std::cout << "Grid written to grid-1_reset_manifold.vtk" << std::endl;

    triangulation.refine_global(i);

  }

  //for(auto& cell: triangulation.active_cell_iterators())
  //  std::cout<<"Cell "<<cell<<" , "<<cell->center()<<std::endl;

  //AssertDimension(triangulation.n_active_quads(),
  //                triangulation.n_active_cells());

  /*std::ofstream out("grid-1_reset_manifold.vtk");
  GridOut       grid_out;
  grid_out.write_vtk(triangulation, out);
  std::cout << "Grid written to grid-1_reset_manifold.vtk" << std::endl;
  print_info(triangulation);*/
}


void second_grid()
{
  Triangulation<2> triangulation;

  const Point<2> center(1, 0);
  const double   inner_radius = 0.5, outer_radius = 1.0;
  GridGenerator::hyper_shell(
    triangulation, center, inner_radius, outer_radius, 10);
  triangulation.reset_all_manifolds();

  for (unsigned int step = 0; step < 5; ++step)
    {
      for (auto &cell : triangulation.active_cell_iterators())
        {
          for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
            {
              const double distance_from_center =
                center.distance(cell->vertex(v));

              if (std::fabs(distance_from_center - inner_radius) < 1e-10)
                {
                  cell->set_refine_flag();
                  break;
                }
            }
        }

      triangulation.execute_coarsening_and_refinement();
    }


  std::ofstream out("grid-2_reset_manifold.svg");
  GridOut       grid_out;
  grid_out.write_svg(triangulation, out);

  std::cout << "Grid written to grid-2_reset_manifold.svg" << std::endl;
  print_info(triangulation);

}

void third_grid() {
    const int dim = 2;

    Triangulation<dim> triangulation;
    const Point<dim> center(1, 0);
    const double radius = 1.0;
    const unsigned int ID = 100;

    GridGenerator::hyper_ball(triangulation,center,radius,true);
    triangulation.reset_all_manifolds();
    SphericalManifold<2> manifold(center);
    triangulation.set_all_manifold_ids(ID);

    /*
    for(auto& cell:triangulation.active_cell_iterators()) {
        for(unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell;++f) {
            if(cell->face(f)->at_boundary())
                cell->face(f)->set_manifold_id(100);
        }
    }
    */


    triangulation.set_manifold(ID,manifold);

    const unsigned int ref_level = 2;

    std::ofstream out_glob("grid-3_"+std::to_string(0)+".vtk");
    GridOut       grid_out;
    grid_out.write_vtk(triangulation, out_glob);

    for(unsigned int i = 1; i <= ref_level; ++i) {
        triangulation.refine_global(i);

        std::ofstream out("grid-3_"+std::to_string(i)+".vtk");
        grid_out.write_vtk(triangulation, out);
        //std::cout << "Grid written to grid-1_reset_manifold.vtk" << std::endl;
    }

}

void fourth_grid() {
    const int dim = 2;

    Triangulation<dim> triangulation;

    const double left = -1;
    const double right = 1;
    GridGenerator::hyper_L(triangulation,left,right,true);

    triangulation.refine_global();
    std::ofstream out("grid-4_global_refine.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(triangulation, out);

    const unsigned int local_ref = 5;
    const Point<dim> corner((left+right)/2,(left+right)/2);

    for(unsigned int i = 0; i < local_ref; ++i) {

        for(auto& cell:triangulation.active_cell_iterators()) {

            for(unsigned v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v) {

                const double distance_from_corner = corner.distance(cell->vertex(v));

                if(std::fabs(distance_from_corner) < 0.3) {
                  cell->set_refine_flag();
                  break;
                }
            }
        }

        triangulation.execute_coarsening_and_refinement();
    }

    std::ofstream out_local("grid-4_local_refine.vtk");
    grid_out.write_vtk(triangulation, out_local);

}


int main()
{
  //first_grid();
  //second_grid();
  third_grid();
  //fourth_grid();
}
