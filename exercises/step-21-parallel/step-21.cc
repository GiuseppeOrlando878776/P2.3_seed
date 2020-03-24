/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2006 - 2019 by the deal.II authors
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

 *
 * Authors: Yan Li, Wolfgang Bangerth, Texas A&M University, 2006
 */


// This program is an adaptation of step-20 and includes some technique of DG
// methods from step-12. A good part of the program is therefore very similar
// to step-20 and we will not comment again on these parts. Only the new stuff
// will be discussed in more detail.

// @sect3{Include files}

// All of these include files have been used before:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

// In this program, we use a tensor-valued coefficient. Since it may have a
// spatial dependence, we consider it a tensor-valued function. The following
// include file provides the <code>TensorFunction</code> class that offers
// such functionality:
#include <deal.II/base/tensor_function.h>

// Include for TimerOutput
#include <deal.II/base/timer.h>

// Include for parameter handler
#include <deal.II/base/parameter_handler.h>

// Includes for parallel version
#include <deal.II/base/mpi.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/filtered_iterator.h>

#include <deal.II/lac/generic_linear_algebra.h>

// Includes for multithread computations
#include <deal.II/base/work_stream.h>
#include <deal.II/meshworker/scratch_data.h>
#include <deal.II/meshworker/copy_data.h>

// The last step is as in all previous programs:
namespace Step21 {
  using namespace dealii;

  namespace LA {
     //using namespace dealii::LinearAlgebraPETSc;
     using namespace dealii::LinearAlgebraTrilinos;
  }


  // @sect3{The <code>TwoPhaseFlowProblem</code> class}

  // This is the main class of the program. It is close to the one of step-20,
  // but with a few additional functions:
  //
  // <ul> <li><code>assemble_rhs_S</code> assembles the right hand side of the
  //   saturation equation. As explained in the introduction, this can't be
  //   integrated into <code>assemble_rhs</code> since it depends on the
  //   velocity that is computed in the first part of the time step.
  //
  //   <li><code>get_maximal_velocity</code> does as its name suggests. This
  //   function is used in the computation of the time step size.
  //
  //   <li><code>project_back_saturation</code> resets all saturation degrees
  //   of freedom with values less than zero to zero, and all those with
  //   saturations greater than one to one.  </ul>
  //
  // The rest of the class should be pretty much obvious. The
  // <code>viscosity</code> variable stores the viscosity $\mu$ that enters
  // several of the formulas in the nonlinear equations.
  template<int dim>
  class TwoPhaseFlowProblem {
  public:
    struct Parameters;    /*! \brief Forward declaration of the class to handle some parameters */
    TwoPhaseFlowProblem(const std::string& parameter_filename);
    void run();
    using CellFilter = FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  protected:
    std::shared_ptr<Parameters> parameters; /*! \brief Variable of the class to handle parameters */

  private:
    void   make_grid();

    void   setup_dofs();
    void   setup_dofs_initial_condition();  /*! \brief Auxiliary function to set dofs for projection of initial condition */

    void   assemble_initial_condition_one_cell(const typename DoFHandler<dim>::active_cell_iterator& cell,
                                               MeshWorker::ScratchData<dim>& scratch,
                                               MeshWorker::CopyData<>& data);   /*! \brief Auxiliary function to assemble
                                                                                          the projection system on
                                                                                          one cell */
    void   copy_local_to_global_initial_condition(const MeshWorker::CopyData<>& data); /*! \brief Copy globally system for
                                                                                                  initial condition
                                                                                                  projection */
    void   assemble_initial_condition();  /*! \brief Auxiliary function to assemble the projection system
                                                    for initial condition */

    void   solve_initial_condition();    /*! \brief Auxiliary function to solve the projection system for initial data */

    void   assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator& cell,
                                    MeshWorker::ScratchData<dim>& scratch,
                                    MeshWorker::CopyData<>& data); /*! \brief Assemble system matrix and rhs of pressure
                                                                              and velocity on single cell */
    void   copy_local_to_global_system(const MeshWorker::CopyData<>& data); /*! \brief Copy globally system matrix and
                                                                                       rhs of pressure and velocity
                                                                                       on single cell */
    void   assemble_system();
    void   assemble_rhs_S_one_cell(const typename DoFHandler<dim>::active_cell_iterator& cell,
                                    MeshWorker::ScratchData<dim>& scratch,
                                    MeshWorker::CopyData<>& data);  /*! \brief Assemble saturation rhs
                                                                               on single cell */
    void   copy_local_to_global_rhs_S(const MeshWorker::CopyData<>& data); /*! \brief Copy globally saturation rhs
                                                                                      on single cell */
    void   assemble_rhs_S();

    double get_maximal_velocity() const;
    void   solve();
    void   project_back_saturation();
    void   output_results() const;

    MPI_Comm  communicator;   /*! \brief Communicator for parallel distributed memory */

    const unsigned int degree;

    parallel::distributed::Triangulation<dim> triangulation;
    FESystem<dim>                             fe;
    DoFHandler<dim>                           dof_handler;
    DoFHandler<dim>                           tmp_dof_handler;  /*! \brief Auxiliary DofHandler for initial
                                                                          condition projection */

    LA::MPI::SparseMatrix initial_matrix; /*! \brief Auxiliary matrix for initial condition projection */
    LA::MPI::Vector initial_rhs;          /*! \brief Auxiliary vector for initial condition projection */

    IndexSet   locally_owned_dofs,    /*! \brief Set of locally owned dofs */
               locally_relevant_dofs; /*! \brief Set of locally relevant dofs */

    std::vector<IndexSet> locally_owned_dofs_partition,     /*! \brief Partition between variables of locally owned dofs */
                          locally_relevant_dofs_partition;  /*! \brief Partition between variables of locally relevant dofs */

    LA::MPI::BlockVector effective_solution;  /*! \brief Auxiliary vector with only owned dofs for solution */

    LA::MPI::BlockSparseMatrix system_matrix;

    const unsigned int n_refinement_steps;

    double       time_step;
    unsigned int timestep_number;
    double       end_time;                                /*! \brief End time of simulation */
    double       viscosity;

    LA::MPI::BlockVector solution;
    LA::MPI::BlockVector old_solution;
    LA::MPI::BlockVector system_rhs;

    LA::MPI::BlockVector locally_relevant_solution;    /*! \brief BlockVector to store locally relevant solution */

    ConditionalOStream  pcout;       /*! \brief Output stream only for rank 0 */

    std::ofstream       time_out;     /*!  \brief Auxiliary ofstream for time output */
    ConditionalOStream  ptime_out;    /*!  \brief Auxiliary conditional stream for time output */
    TimerOutput         time_table;   /*! \brief Table for time */

  public:
    struct Parameters {
      Parameters(const std::string& parameter_filename);  /*! \brief Class constructor with the name of the file */

      static void declare_parameters(ParameterHandler& prm);   /*! \brief Declaration of parameters using ParameterHandler.
                                                                          It is static because it is associated to the class
                                                                          and not to a particular instance. */
      void parse_parameters(ParameterHandler& prm);            /*! \brief Pase function with auxilium of ParameterHandler */

      unsigned int degree;    /*! \brief Polynomial degree of FE spaces */

      double end_time;  /*! \brief End time of simulation */

      unsigned int n_global_refinement; /*! \brief Number of global refinement */

      double viscosity;  /*! \brief Viscosity of the flow */

      std::string time_name;  /*! \brief File name of auxiliary stream for output of time */
    };

  };


  // @sect3{Equation data}

  // @sect4{Pressure right hand side}

  // At present, the right hand side of the pressure equation is simply the
  // zero function. However, the rest of the program is fully equipped to deal
  // with anything else, if this is desired:
  template<int dim>
  class PressureRightHandSide : public Function<dim> {
  public:
    PressureRightHandSide() : Function<dim>(1) {}

    virtual double value(const Point<dim> & /*p*/,
                         const unsigned int /*component*/ = 0) const override {
      return 0;
    }
  };



  // @sect4{Pressure boundary values}

  // The next are pressure boundary values. As mentioned in the introduction,
  // we choose a linear pressure field:
  template<int dim>
  class PressureBoundaryValues : public Function<dim> {
  public:
    PressureBoundaryValues() : Function<dim>(1) {}

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override {
      return 1 - p[0];
    }
  };



  // @sect4{Saturation boundary values}

  // Then we also need boundary values on the inflow portions of the
  // boundary. The question whether something is an inflow part is decided
  // when assembling the right hand side, we only have to provide a functional
  // description of the boundary values. This is as explained in the
  // introduction:
  template<int dim>
  class SaturationBoundaryValues : public Function<dim> {
  public:
    SaturationBoundaryValues() : Function<dim>(1) {}

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override {
      if (p[0] == 0)
        return 1;
      else
        return 0;
    }
  };



  // @sect4{Initial data}

  // Finally, we need initial data. In reality, we only need initial data for
  // the saturation, but we are lazy, so we will later, before the first time
  // step, simply interpolate the entire solution for the previous time step
  // from a function that contains all vector components.
  //
  // We therefore simply create a function that returns zero in all
  // components. We do that by simply forward every function to the
  // Functions::ZeroFunction class. Why not use that right away in the places of
  // this program where we presently use the <code>InitialValues</code> class?
  // Because this way it is simpler to later go back and choose a different
  // function for initial values.
  template<int dim>
  class InitialValues : public Function<dim> {
  public:
    InitialValues() : Function<dim>(dim + 2) {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override {
      return Functions::ZeroFunction<dim>(dim + 2).value(p, component);
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  values) const override {
      Functions::ZeroFunction<dim>(dim + 2).vector_value(p, values);
    }
  };



  // @sect3{The inverse permeability tensor}

  // As announced in the introduction, we implement two different permeability
  // tensor fields. Each of them we put into a namespace of its own, so that
  // it will be easy later to replace use of one by the other in the code.

  // @sect4{Single curving crack permeability}

  // The first function for the permeability was the one that models a single
  // curving crack. It was already used at the end of step-20, and its
  // functional form is given in the introduction of the present tutorial
  // program. As in some previous programs, we have to declare a (seemingly
  // unnecessary) default constructor of the KInverse class to avoid warnings
  // from some compilers:
  namespace SingleCurvingCrack {
    template<int dim>
    class KInverse : public TensorFunction<2, dim> {
    public:
      KInverse() : TensorFunction<2, dim>() {}

      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<2, dim>> &  values) const override {

        Assert(points.size() == values.size(),
               ExcDimensionMismatch(points.size(), values.size()));

        for(unsigned int p = 0; p < points.size(); ++p) {
            values[p].clear();

            const double distance_to_flowline = std::fabs(points[p][1] - 0.5 - 0.1 * std::sin(10 * points[p][0]));

            const double permeability = std::max(std::exp(-(distance_to_flowline * distance_to_flowline)/(0.1 * 0.1)), 0.01);

            for(unsigned int d = 0; d < dim; ++d)
              values[p][d][d] = 1. / permeability;
        }
      }
    };
  } // namespace SingleCurvingCrack


  // @sect4{Random medium permeability}

  // This function does as announced in the introduction, i.e. it creates an
  // overlay of exponentials at random places. There is one thing worth
  // considering for this class. The issue centers around the problem that the
  // class creates the centers of the exponentials using a random function. If
  // we therefore created the centers each time we create an object of the
  // present type, we would get a different list of centers each time. That's
  // not what we expect from classes of this type: they should reliably
  // represent the same function.
  //
  // The solution to this problem is to make the list of centers a static
  // member variable of this class, i.e. there exists exactly one such
  // variable for the entire program, rather than for each object of this
  // type. That's exactly what we are going to do.
  //
  // The next problem, however, is that we need a way to initialize this
  // variable. Since this variable is initialized at the beginning of the
  // program, we can't use a regular member function for that since there may
  // not be an object of this type around at the time. The C++ standard
  // therefore says that only non-member and static member functions can be
  // used to initialize a static variable. We use the latter possibility by
  // defining a function <code>get_centers</code> that computes the list of
  // center points when called.
  //
  // Note that this class works just fine in both 2d and 3d, with the only
  // difference being that we use more points in 3d: by experimenting we find
  // that we need more exponentials in 3d than in 2d (we have more ground to
  // cover, after all, if we want to keep the distance between centers roughly
  // equal), so we choose 40 in 2d and 100 in 3d. For any other dimension, the
  // function does presently not know what to do so simply throws an exception
  // indicating exactly this.
  namespace RandomMedium {
    template<int dim>
    class KInverse : public TensorFunction<2, dim> {
    public:
      KInverse() : TensorFunction<2, dim>() {}

      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<2, dim>> &  values) const override {

        Assert(points.size() == values.size(),
               ExcDimensionMismatch(points.size(), values.size()));

        for(unsigned int p = 0; p < points.size(); ++p) {
            values[p].clear();

            double permeability = 0;
            for(unsigned int i = 0; i < centers.size(); ++i)
              permeability += std::exp(-(points[p] - centers[i]).norm_square()/(0.05 * 0.05));

            const double normalized_permeability = std::min(std::max(permeability, 0.01), 4.);

            for(unsigned int d = 0; d < dim; ++d)
              values[p][d][d] = 1. / normalized_permeability;
        }
      }

    private:
      static std::vector<Point<dim>> centers;

      static std::vector<Point<dim>> get_centers() {
        const unsigned int N = (dim == 2 ? 40 : (dim == 3 ? 100 : throw ExcNotImplemented()));

        std::vector<Point<dim>> centers_list(N);
        for(unsigned int i = 0; i < N; ++i)
          for(unsigned int d = 0; d < dim; ++d)
            centers_list[i][d] = static_cast<double>(rand()) / RAND_MAX;

        return centers_list;
      }
    };


    template<int dim>
    std::vector<Point<dim>> KInverse<dim>::centers = KInverse<dim>::get_centers();
  } // namespace RandomMedium



  // @sect3{The inverse mobility and saturation functions}

  // There are two more pieces of data that we need to describe, namely the
  // inverse mobility function and the saturation curve. Their form is also
  // given in the introduction:
  double mobility_inverse(const double S, const double viscosity) {
    return 1.0/((1.0/viscosity)*S*S + (1 - S)*(1 - S));
  }

  double fractional_flow(const double S, const double viscosity) {
    return S*S/(S*S + viscosity*(1 - S)*(1 - S));
  }



  // @sect3{Linear solvers and preconditioners}

  // The linear solvers we use are also completely analogous to the ones used
  // in step-20. The following classes are therefore copied verbatim from
  // there. Note that the classes here are not only copied from
  // step-20, but also duplicate classes in deal.II. In a future version of this
  // example, they should be replaced by an efficient method, though. There is a
  // single change: if the size of a linear system is small, i.e. when the mesh
  // is very coarse, then it is sometimes not sufficient to set a maximum of
  // <code>src.size()</code> CG iterations before the solver in the
  // <code>vmult()</code> function converges. (This is, of course, a result of
  // numerical round-off, since we know that on paper, the CG method converges
  // in at most <code>src.size()</code> steps.) As a consequence, we set the
  // maximum number of iterations equal to the maximum of the size of the linear
  // system and 200.
  template<class MatrixType, class PrecondType = PreconditionIdentity>
  class InverseMatrix : public Subscriptor {
  public:
    static_assert(std::is_default_constructible<PrecondType>::value,
                  "Preconditioner type not default constructable");

    InverseMatrix(const MatrixType& m, const PrecondType& p = PrecondType()) : matrix(&m), preconditioner(p) {}

    void vmult(LA::MPI::Vector& dst, const LA::MPI::Vector& src) const {
      SolverControl solver_control(std::max<unsigned int>(src.size(), 200),
                                   1e-8 * src.l2_norm());
      SolverCG<LA::MPI::Vector>   cg(solver_control);

      dst = 0;

      cg.solve(*matrix, dst, src, preconditioner);
    }

  private:
    const SmartPointer<const MatrixType> matrix;
    const PrecondType& preconditioner;
  };



  class SchurComplement : public Subscriptor {
  public:
    SchurComplement(const LA::MPI::BlockSparseMatrix&           A,
                    const InverseMatrix<LA::MPI::SparseMatrix>& Minv,
                    const IndexSet& local_owned_vel,
                    const MPI_Comm& mpi_comm) :
      system_matrix(&A),
      m_inverse(&Minv),
      tmp1(local_owned_vel, mpi_comm),
      tmp2(tmp1)
      {}

    void vmult(LA::MPI::Vector& dst, const LA::MPI::Vector& src) const {
      system_matrix->block(0, 1).vmult(tmp1, src);
      m_inverse->vmult(tmp2, tmp1);
      system_matrix->block(1, 0).vmult(dst, tmp2);
    }

  private:
    const SmartPointer<const LA::MPI::BlockSparseMatrix>           system_matrix;
    const SmartPointer<const InverseMatrix<LA::MPI::SparseMatrix>> m_inverse;

    mutable LA::MPI::Vector tmp1, tmp2;
  };



  class ApproximateSchurComplement : public Subscriptor {
  public:
    ApproximateSchurComplement(const LA::MPI::BlockSparseMatrix& A,
                               const IndexSet& local_owned_vel,
                               const MPI_Comm& mpi_comm) :
      system_matrix(&A),
      tmp1(local_owned_vel, mpi_comm),
      tmp2(tmp1)
      {}

    void vmult(LA::MPI::Vector& dst, const LA::MPI::Vector& src) const {
      system_matrix->block(0, 1).vmult(tmp1, src);
      auto tmp_Jacobi_precond = LA::MPI::PreconditionJacobi();
      tmp_Jacobi_precond.initialize(system_matrix->block(0,0));
      InverseMatrix<LA::MPI::SparseMatrix,
                    decltype(tmp_Jacobi_precond)> tmp_matrix(system_matrix->block(0,0), tmp_Jacobi_precond);
      tmp_matrix.vmult(tmp2,tmp1);
      //system_matrix->block(0, 0).precondition_Jacobi(tmp2, tmp1);
      system_matrix->block(1, 0).vmult(dst, tmp2);
    }

  private:
    const SmartPointer<const LA::MPI::BlockSparseMatrix> system_matrix;

    mutable LA::MPI::Vector tmp1, tmp2;
  };



  // @sect3{<code>TwoPhaseFlowProblem</code> class implementation}

  // Here now the implementation of the main class. Much of it is actually
  // copied from step-20, so we won't comment on it in much detail. You should
  // try to get familiar with that program first, then most of what is
  // happening here should be mostly clear.

  // First the constructor of the Parameters struct
  template<int dim>
  TwoPhaseFlowProblem<dim>::Parameters::Parameters(const std::string& parameter_filename):
      degree(0),
      end_time(1.0),
      n_global_refinement(5),
      viscosity(0.2),
      time_name("Time_analysis") {

        ParameterHandler prm;
        TwoPhaseFlowProblem<dim>::Parameters::declare_parameters(prm);

        std::ifstream parameter_file(parameter_filename);

        if(!parameter_file.is_open()) {
          std::ofstream parameter_out(parameter_filename);
          prm.print_parameters(parameter_out, ParameterHandler::Text);

          std::cerr<<"Input parameter file <" + parameter_filename +
                     "> not found. Creating a default file."<<std::endl;

          parameter_file.open(parameter_filename);
        }

        prm.parse_input(parameter_file);
        parse_parameters(prm);
      }



  // Next we have a function that declares the parameters that we expect in
  // the input file, together with their data types, default values and a
  // description:
  template <int dim>
  void TwoPhaseFlowProblem<dim>::Parameters::declare_parameters(ParameterHandler &prm) {
    prm.declare_entry("Degree", "0", Patterns::Integer(0), "Polynomial degree of FE spaces");
    prm.declare_entry("End time", "1.0", Patterns::Double(0.0), "End time of the simulation.");
    prm.declare_entry("Number global refinements", "5", Patterns::Integer(0), "Number of global refinements");
    prm.declare_entry("Viscosity", "0.2", Patterns::Double(0.0), "Viscosity");
    prm.declare_entry("Name file time", "Time_analysis", Patterns::Anything(), "Name of the file for time table");
  }


  // And then we need a function that reads the contents of the
  // ParameterHandler object we get by reading the input file and puts the
  // results into variables that store the values of the parameters we have
  // previously declared:
  template <int dim>
  void TwoPhaseFlowProblem<dim>::Parameters::parse_parameters(ParameterHandler &prm) {
    this->degree              = prm.get_integer("Degree");
    this->end_time            = prm.get_double("End time");
    this->n_global_refinement = prm.get_integer("Number global refinements");
    this->viscosity           = prm.get_double("Viscosity");
    this->time_name           = prm.get("Name file time");
  }


  // @sect4{TwoPhaseFlowProblem::TwoPhaseFlowProblem}

  // First for the constructor. We use $RT_k \times DQ_k \times DQ_k$
  // spaces. The time step is set to zero initially, but will be computed
  // before it is needed first, as described in a subsection of the
  // introduction.
  template<int dim>
  TwoPhaseFlowProblem<dim>::TwoPhaseFlowProblem(const std::string& parameter_filename) :
    parameters(std::make_shared<Parameters>(parameter_filename)),
    communicator(MPI_COMM_WORLD),
    degree(parameters->degree),
    triangulation(communicator),
    fe(FE_RaviartThomas<dim>(degree),
       1,
       FE_DGQ<dim>(degree),
       1,
       FE_DGQ<dim>(degree),
       1),
    dof_handler(triangulation),
    tmp_dof_handler(triangulation),
    n_refinement_steps(parameters->n_global_refinement),
    time_step(0),
    timestep_number(1),
    end_time(parameters->end_time),
    viscosity(parameters->viscosity),
    pcout(std::cout, Utilities::MPI::this_mpi_process(communicator) == 0),
    time_out(parameters->time_name + "_" + Utilities::int_to_string(Utilities::MPI::n_mpi_processes(communicator)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(communicator) == 0),
    time_table(communicator, ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times) {
      //Reserve capacity for local partitions
      locally_owned_dofs_partition.reserve(3);
      locally_relevant_dofs_partition.reserve(3);

    }


  // @sect4{TwoPhaseFlowProblem::make_grid}

  // This next function starts out with well-known functions calls that create
  // and refine a mesh, and then associate degrees of freedom with it. It does
  // all the same things as in step-20, just now for three components instead
  // of two.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::make_grid() {
    TimerOutput::Scope t(time_table, "Make grid");

    GridGenerator::hyper_cube(triangulation, 0, 1);
    triangulation.refine_global(n_refinement_steps);
  }

  // @sect4{TwoPhaseFlowProblem::setup_dofs}

  // This next function starts associates degrees of freedom with triangulation. It does
  // all the same things as in step-20, just now for three components instead
  // of two.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::setup_dofs() {
    TimerOutput::Scope t(time_table, "Setup Dofs");

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);

    std::vector<types::global_dof_index> dofs_per_component(dim + 2);
    DoFTools::count_dofs_per_component(dof_handler, dofs_per_component);
    const unsigned int n_u = dofs_per_component[0],
                       n_p = dofs_per_component[dim],
                       n_s = dofs_per_component[dim + 1];

    pcout << "Number of active cells: " << triangulation.n_global_active_cells()
          << std::endl
          << "Number of degrees of freedom: "
          << " (" << n_u << '+' << n_p << '+' << n_s << ')' << std::endl
          << std::endl;

    locally_owned_dofs = dof_handler.locally_owned_dofs();

    // Clear for safety
    locally_owned_dofs_partition.clear();

    locally_owned_dofs_partition.push_back(locally_owned_dofs.get_view(0,n_u));
    locally_owned_dofs_partition.push_back(locally_owned_dofs.get_view(n_u,n_u + n_p));
    locally_owned_dofs_partition.push_back(locally_owned_dofs.get_view(n_u + n_p, n_u + n_p + n_s));

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Clear for safety
    locally_relevant_dofs_partition.clear();

    locally_relevant_dofs_partition.push_back(locally_relevant_dofs.get_view(0,n_u));
    locally_relevant_dofs_partition.push_back(locally_relevant_dofs.get_view(n_u,n_u + n_p));
    locally_relevant_dofs_partition.push_back(locally_relevant_dofs.get_view(n_u + n_p, n_u + n_p + n_s));

    //We erase now the second component of the dofs_per_component since the call
    //'count_dofs_per_component' computes the dofs for RT space as two components
    //but in reality it is vectorial and so the dofs of one component are the info we want
    dofs_per_component.erase(dofs_per_component.begin() + 1);

    BlockDynamicSparsityPattern dsp(dofs_per_component, dofs_per_component);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs_per_processor(),
                                               communicator, locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs_partition, dsp, communicator);
    system_matrix.collect_sizes();

    solution.reinit(locally_relevant_dofs_partition, communicator);
    solution.collect_sizes();

    old_solution.reinit(solution);
    old_solution.collect_sizes();

    effective_solution.reinit(locally_owned_dofs_partition, communicator);
    effective_solution.collect_sizes();

    system_rhs.reinit(locally_owned_dofs_partition, communicator);
    system_rhs.collect_sizes();

    locally_relevant_solution.reinit(locally_owned_dofs_partition,
                                     locally_relevant_dofs_partition,
                                     communicator);
  }

  // @sect4{TwoPhaseFlowProblem::setup_dofs_initial_condition}

  // This is the function that sets the dofs to project the initial
  // condition in parallel. Since the only time dependent variable is the saturation
  // we choose to build a new DofHandler object to deal only with the dofs related to
  // that variable. In this way we pay something at level of construction but we are
  // much more efficient in the resolution of the system

  template<int dim>
  void TwoPhaseFlowProblem<dim>::setup_dofs_initial_condition() {
    TimerOutput::Scope t(time_table, "Setup dofs initial_condition");

    // Extract the corresponding finite element space
    const auto& tmp_fe = fe.get_sub_fe(dim + 1, 1);

    // Build the new dof handler
    tmp_dof_handler.distribute_dofs(tmp_fe);
    DoFRenumbering::Cuthill_McKee(tmp_dof_handler);

    // Build and distribute the sparsity pattern associated to this
    // subspace
    auto& initially_owned_dofs = tmp_dof_handler.locally_owned_dofs();
    IndexSet initially_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(tmp_dof_handler, initially_relevant_dofs);

    DynamicSparsityPattern tmp_dsp(tmp_dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(tmp_dof_handler, tmp_dsp);
    SparsityTools::distribute_sparsity_pattern(tmp_dsp, tmp_dof_handler.n_locally_owned_dofs_per_processor(),
                                               communicator, initially_relevant_dofs);

    // Initialize matrices and vectors to store system for the projection of
    // the initial condition
    initial_matrix.reinit(initially_owned_dofs, tmp_dsp, communicator);
    initial_rhs.reinit(initially_owned_dofs, communicator);
  }

  // @sect4{TwoPhaseFlowProblem::assemble_initial_condition_one_cell}

  // This is the function that assembles the linear system to project the initial
  // condition in parallel for one cell. We have to deal with data only for saturation
  // to solve (\phi_i_s,phi_i_j)F_j = (\phi_i_s,f), where f is the initial codition.

  template<int dim>
  void TwoPhaseFlowProblem<dim>::assemble_initial_condition_one_cell(const typename DoFHandler<dim>::active_cell_iterator& cell,
                                                                     MeshWorker::ScratchData<dim>& scratch,
                                                                     MeshWorker::CopyData<>& data) {
    // Build the linear system to be solved
    auto& fe_values = scratch.reinit(cell);

    const unsigned int dofs_per_cell = fe.get_sub_fe(dim + 1, 1).dofs_per_cell;

    const unsigned int n_q_points    = fe_values.n_quadrature_points;

    const InitialValues<dim> initial_condition;

    std::vector<double> rhs_values;
    rhs_values.reserve(n_q_points);

    data.matrices[0] = 0;
    data.vectors[0]  = 0;

    // Then we have to get the values of the initial condition:
    for(const auto& actual_q_point: fe_values.get_quadrature_points())
      rhs_values.emplace_back(initial_condition.value(actual_q_point, dim + 1));

    // With all this, we can now loop over all the quadrature points and
    // shape functions on this cell and assemble those parts of the matrix
    // and right hand side that we deal with in this function.
    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
            data.matrices[0](i, j) +=  (fe_values.shape_value(i, q) *
                                        fe_values.shape_value(j, q)) *
                                        fe_values.JxW(q);
          }
          data.vectors[0](i) += (fe_values.shape_value(i, q) * rhs_values[q]) * fe_values.JxW(q);
      }
    }

    // The final step is the loop over all cells is to transfer local
    // contributions into the global matrix and right hand side vector:
    // therefore we save dof__indices
    cell->get_dof_indices(data.local_dof_indices[0]);
  }


  // @sect4{TwoPhaseFlowProblem::copy_local_to_global_initial_condition}

  // This is the function that copies globally the matrix related to the initially
  // condition.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::copy_local_to_global_initial_condition(const MeshWorker::CopyData<>& data) {
    const unsigned int dofs_per_cell = fe.get_sub_fe(dim + 1,1).dofs_per_cell;

    for(unsigned int i = 0; i < dofs_per_cell; ++i) {
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        initial_matrix.add(data.local_dof_indices[0][i],
                           data.local_dof_indices[0][j],
                           data.matrices[0](i, j));
      initial_rhs(data.local_dof_indices[0][i]) += data.vectors[0](i);
    }
  }


  // @sect4{TwoPhaseFlowProblem::assemble_initial_condition}

  // This is the function that assembles the linear system to project the initial
  // condition in multithread parallel.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::assemble_initial_condition() {
    TimerOutput::Scope t(time_table, "Assemble initial_condition");

    initial_matrix = 0;
    initial_rhs = 0;

    WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(), tmp_dof_handler.begin_active()),
                    CellFilter(IteratorFilters::LocallyOwnedCell(), tmp_dof_handler.end()),
                    std::bind(&TwoPhaseFlowProblem<dim>::assemble_initial_condition_one_cell,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                    std::bind(&TwoPhaseFlowProblem<dim>::copy_local_to_global_initial_condition,
                              this, std::placeholders::_1),
                    MeshWorker::ScratchData<dim>(fe.get_sub_fe(dim + 1,1),
                                                 QGauss<dim>(degree + 2),
                                                 update_values | update_quadrature_points | update_JxW_values
                                                 ),
                    MeshWorker::CopyData(fe.get_sub_fe(dim + 1,1).dofs_per_cell));

    initial_matrix.compress(VectorOperation::add);
    initial_rhs.compress(VectorOperation::add);
  }


  // @sect4{TwoPhaseFlowProblem::solve_initial_condition}

  // This is the function that solves the linear system to project the initial
  // condition in parallel.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::solve_initial_condition() {
    TimerOutput::Scope t(time_table, "Solve initial_condition");

    SolverControl solver_control(effective_solution.block(2).size(),
                                 1e-12 * initial_rhs.l2_norm());
    SolverCG<LA::MPI::Vector>  cg(solver_control);
    cg.solve(initial_matrix, effective_solution.block(2), initial_rhs, PreconditionIdentity());

    old_solution = effective_solution;
  }


  // @sect4{TwoPhaseFlowProblem::assemble_system_one_cell}

  // This is the function that assembles the linear system on one cell, or at least
  // everything except the (1,3) block that depends on the still-unknown
  // velocity computed during this time step (we deal with this in
  // <code>assemble_rhs_S</code>). Much of it is again as in step-20, but we
  // have to deal with some nonlinearity this time.  However, the top of the
  // function is pretty much as usual (note that we set matrix and right hand
  // side to zero at the beginning &mdash; something we didn't have to do for
  // stationary problems since there we use each matrix object only once and
  // it is empty at the beginning anyway).
  //
  // Note that in its present form, the function uses the permeability
  // implemented in the RandomMedium::KInverse class. Switching to the single
  // curved crack permeability function is as simple as just changing the
  // namespace name.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::assemble_system_one_cell(const typename DoFHandler<dim>::active_cell_iterator& cell,
                                                          MeshWorker::ScratchData<dim>& scratch,
                                                          MeshWorker::CopyData<>& data) {

    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    auto& fe_values = scratch.reinit(cell);

    const unsigned int n_q_points    = fe_values.n_quadrature_points;

    const PressureRightHandSide<dim>  pressure_right_hand_side;
    const PressureBoundaryValues<dim> pressure_boundary_values;
    const RandomMedium::KInverse<dim> k_inverse;

    std::vector<double>         pressure_rhs_values(n_q_points);
    std::vector<Tensor<2, dim>> k_inverse_values(n_q_points);

    std::vector<Vector<double>>              old_solution_values(n_q_points, Vector<double>(dim + 2));
    std::vector<std::vector<Tensor<1, dim>>> old_solution_grads(n_q_points, std::vector<Tensor<1, dim>>(dim + 2));

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Scalar saturation(dim + 1);

    data.matrices[0] = 0;
    data.vectors[0]  = 0;

    // Here's the first significant difference: We have to get the values
    // of the saturation function of the previous time step at the
    // quadrature points. To this end, we can use the
    // FEValues::get_function_values (previously already used in step-9,
    // step-14 and step-15), a function that takes a solution vector and
    // returns a list of function values at the quadrature points of the
    // present cell. In fact, it returns the complete vector-valued
    // solution at each quadrature point, i.e. not only the saturation but
    // also the velocities and pressure:
    fe_values.get_function_values(old_solution, old_solution_values);

    // Then we also have to get the values of the pressure right hand side
    // and of the inverse permeability tensor at the quadrature points:
    pressure_right_hand_side.value_list(fe_values.get_quadrature_points(),
                                        pressure_rhs_values);
    k_inverse.value_list(fe_values.get_quadrature_points(),
                         k_inverse_values);

    // With all this, we can now loop over all the quadrature points and
    // shape functions on this cell and assemble those parts of the matrix
    // and right hand side that we deal with in this function. The
    // individual terms in the contributions should be self-explanatory
    // given the explicit form of the bilinear form stated in the
    // introduction:
    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double old_s = old_solution_values[q](dim + 1);

          const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
          const double div_phi_i_u     = fe_values[velocities].divergence(i, q);
          const double phi_i_p         = fe_values[pressure].value(i, q);
          const double phi_i_s         = fe_values[saturation].value(i, q);

          for(unsigned int j = 0; j < dofs_per_cell; ++j) {
              const Tensor<1, dim> phi_j_u = fe_values[velocities].value(j, q);
              const double div_phi_j_u     = fe_values[velocities].divergence(j, q);
              const double phi_j_p         = fe_values[pressure].value(j, q);
              const double phi_j_s         = fe_values[saturation].value(j, q);

              data.matrices[0](i, j) += (phi_i_u * k_inverse_values[q] *
                                         mobility_inverse(old_s, viscosity) * phi_j_u -
                                         div_phi_i_u * phi_j_p - phi_i_p * div_phi_j_u +
                                         phi_i_s * phi_j_s) *
                                         fe_values.JxW(q);
          }

          data.vectors[0](i) += (-phi_i_p * pressure_rhs_values[q]) * fe_values.JxW(q);
      }
    }

    // Next, we also have to deal with the pressure boundary values. This,
    // again is as in step-20:
    for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
      if(cell->at_boundary(face_no)) {
        auto& fe_face_values = scratch.reinit(cell, face_no);
        const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;

        std::vector<double>  boundary_values(n_face_q_points);
        pressure_boundary_values.value_list(fe_face_values.get_quadrature_points(), boundary_values);

        for(unsigned int q = 0; q < n_face_q_points; ++q) {
          for(unsigned int i = 0; i < dofs_per_cell; ++i) {
            const Tensor<1, dim> phi_i_u = fe_face_values[velocities].value(i, q);
            data.vectors[0](i) += -(phi_i_u * fe_face_values.normal_vector(q) *
                                    boundary_values[q] * fe_face_values.JxW(q));
          }
        }
      }
    }

    // The final step is the loop over all cells is to transfer local
    // contributions into the global matrix and right hand side vector:
    // therefore we save local_dof_indices
    cell->get_dof_indices(data.local_dof_indices[0]);
  }
  // So much for assembly of matrix and right hand side. Note that we do not
  // have to interpolate and apply boundary values since they have all been
  // taken care of in the weak form already.


  // @sect4{TwoPhaseFlowProblem::copy_local_to_global_system}

  // This is the function that transfers globally the previous
  // assembled linear system
  template<int dim>
  void TwoPhaseFlowProblem<dim>::copy_local_to_global_system(const MeshWorker::CopyData<>& data) {
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
      for(unsigned int j = 0; j < fe.dofs_per_cell; ++j)
        system_matrix.add(data.local_dof_indices[0][i],
                          data.local_dof_indices[0][j],
                          data.matrices[0](i, j));
      system_rhs(data.local_dof_indices[0][i]) += data.vectors[0](i);
    }
  }


  // @sect4{TwoPhaseFlowProblem::assemble_system}

  // This is the function that assembles effectively the linear system using
  // the workstream class
  template<int dim>
  void TwoPhaseFlowProblem<dim>::assemble_system() {
    TimerOutput::Scope t(time_table, "Assemble system");

    system_matrix = 0;
    system_rhs = 0;

    WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.begin_active()),
                    CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.end()),
                    std::bind(&TwoPhaseFlowProblem<dim>::assemble_system_one_cell,
                              this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                    std::bind(&TwoPhaseFlowProblem<dim>::copy_local_to_global_system,
                              this, std::placeholders::_1),
                    MeshWorker::ScratchData<dim>(fe,
                                                 QGauss<dim>(degree + 2),
                                                 update_values | update_gradients |
                                                 update_quadrature_points | update_JxW_values,
                                                 QGauss<dim-1>(degree + 2),
                                                 update_values | update_normal_vectors |
                                                 update_quadrature_points | update_JxW_values
                                                 ),
                    MeshWorker::CopyData(fe.dofs_per_cell));

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }


  // @sect4{TwoPhaseFlowProblem::assemble_rhs_S_one_cell}

  // As explained in the introduction, we can only evaluate the right hand
  // side of the saturation equation once the velocity has been computed. We
  // therefore have this separate function to this end.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::assemble_rhs_S_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                         MeshWorker::ScratchData<dim>& scratch,
                                                         MeshWorker::CopyData<>& data) {
    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    auto& fe_values = scratch.reinit(cell);

    const unsigned int n_q_points = fe_values.n_quadrature_points;

    std::vector<Vector<double>> old_solution_values(n_q_points, Vector<double>(dim + 2));
    std::vector<Vector<double>> present_solution_values(n_q_points, Vector<double>(dim + 2));

    SaturationBoundaryValues<dim> saturation_boundary_values;

    const FEValuesExtractors::Scalar saturation(dim + 1);

    data.vectors[0] = 0;

    fe_values.get_function_values(old_solution, old_solution_values);
    fe_values.get_function_values(solution, present_solution_values);

    // First for the cell terms. These are, following the formulas in the
    // introduction, $(S^n,\sigma)-(F(S^n) \mathbf{v}^{n+1},\nabla
    // \sigma)$, where $\sigma$ is the saturation component of the test
    // function:
    for(unsigned int q = 0; q < n_q_points; ++q) {
      for(unsigned int i = 0; i < dofs_per_cell; ++i) {
          const double   old_s = old_solution_values[q](dim + 1);
          Tensor<1, dim> present_u;
          for(unsigned int d = 0; d < dim; ++d)
            present_u[d] = present_solution_values[q](d);

          const double         phi_i_s = fe_values[saturation].value(i, q);
          const Tensor<1, dim> grad_phi_i_s = fe_values[saturation].gradient(i, q);

          data.vectors[0](i) += (time_step * fractional_flow(old_s, viscosity) *
                                 present_u * grad_phi_i_s +
                                 old_s * phi_i_s) *
                                 fe_values.JxW(q);
      }
    }

    // Secondly, we have to deal with the flux parts on the face
    // boundaries. This was a bit more involved because we first have to
    // determine which are the influx and outflux parts of the cell
    // boundary. If we have an influx boundary, we need to evaluate the
    // saturation on the other side of the face (or the boundary values,
    // if we are at the boundary of the domain).
    //
    // All this is a bit tricky, but has been explained in some detail
    // already in step-9. Take a look there how this is supposed to work!
    for(unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no) {
      auto& fe_face_values = scratch.reinit(cell, face_no);
      unsigned int n_face_q_points = fe_face_values.n_quadrature_points;

      std::vector<Vector<double>> old_solution_values_face(n_face_q_points, Vector<double>(dim + 2));
      // Here we need face values also for current solution; this is way we used effective_solution with
      // the correct number of owned dofs and solution with the relevant dofs
      std::vector<Vector<double>> present_solution_values_face(n_face_q_points, Vector<double>(dim + 2));

      std::vector<double>         neighbor_saturation(n_face_q_points);

      fe_face_values.get_function_values(old_solution, old_solution_values_face);
      fe_face_values.get_function_values(solution, present_solution_values_face);

      if(cell->at_boundary(face_no))
        saturation_boundary_values.value_list(fe_face_values.get_quadrature_points(),
                                              neighbor_saturation);
      else {
        const auto         neighbor = cell->neighbor(face_no);
        const unsigned int neighbor_face = cell->neighbor_of_neighbor(face_no);

        auto& fe_face_values_neighbor = scratch.reinit_neighbor(neighbor, neighbor_face);
        std::vector<Vector<double>> old_solution_values_face_neighbor(n_face_q_points, Vector<double>(dim + 2));
        fe_face_values_neighbor.get_function_values(old_solution, old_solution_values_face_neighbor);

        for(unsigned int q = 0; q < n_face_q_points; ++q)
          neighbor_saturation[q] = old_solution_values_face_neighbor[q](dim + 1);
      }


      for(unsigned int q = 0; q < n_face_q_points; ++q) {
        Tensor<1, dim> present_u_face;
        for(unsigned int d = 0; d < dim; ++d)
          present_u_face[d] = present_solution_values_face[q](d);

        const double normal_flux = present_u_face*fe_face_values.normal_vector(q);

        const bool is_outflow_q_point = (normal_flux >= 0);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          data.vectors[0](i) -= time_step * normal_flux *
                                fractional_flow((is_outflow_q_point == true ?
                                                 old_solution_values_face[q](dim + 1) :
                                                 neighbor_saturation[q]),
                                                 viscosity) *
                                fe_face_values[saturation].value(i, q) *
                                fe_face_values.JxW(q);
      }
    } // End of loop over number of faces

    // The final step in the loop over all cells is to transfer local
    // contributions into the right hand side vector:
    // therefore we save local_dof_indices
    cell->get_dof_indices(data.local_dof_indices[0]);
  }


  // @sect4{TwoPhaseFlowProblem::copy_local_to_global_rhs_S}

  // This is the function that transfers globally the previous
  // assembled right-hand side
  template<int dim>
  void TwoPhaseFlowProblem<dim>::copy_local_to_global_rhs_S(const MeshWorker::CopyData<>& data) {
    for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      system_rhs(data.local_dof_indices[0][i]) += data.vectors[0](i);
  }


  // @sect4{TwoPhaseFlowProblem::assemble_rhs_S}

  // This is the function that assembles effectively the right-hand side of the
  // saturation equation usin WorkStream
  template<int dim>
  void TwoPhaseFlowProblem<dim>::assemble_rhs_S() {
    TimerOutput::Scope t(time_table, "Assemble rhs saturation");

    WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.begin_active()),
                    CellFilter(IteratorFilters::LocallyOwnedCell(), dof_handler.end()),
                    std::bind(&TwoPhaseFlowProblem<dim>::assemble_rhs_S_one_cell,
                              this, std::placeholders::_1, std::placeholders::_2,
                                    std::placeholders::_3),
                    std::bind(&TwoPhaseFlowProblem<dim>::copy_local_to_global_rhs_S,
                              this, std::placeholders::_1),
                    MeshWorker::ScratchData<dim>(fe,
                                                 QGauss<dim>(degree + 2),
                                                 update_values | update_gradients |
                                                 update_quadrature_points | update_JxW_values,
                                                 QGauss<dim-1>(degree + 2),
                                                 update_values | update_normal_vectors |
                                                 update_quadrature_points | update_JxW_values
                                                 ),
                    MeshWorker::CopyData(fe.dofs_per_cell));

    system_rhs.compress(VectorOperation::add);
  }


  // @sect4{TwoPhaseFlowProblem::solve}

  // After all these preparations, we finally solve the linear system for
  // velocity and pressure in the same way as in step-20. After that, we have
  // to deal with the saturation equation (see below):
  template<int dim>
  void TwoPhaseFlowProblem<dim>::solve() {
    TimerOutput::Scope t(time_table, "Solve");

    const InverseMatrix<LA::MPI::SparseMatrix> m_inverse(system_matrix.block(0, 0));
    LA::MPI::Vector tmp(locally_owned_dofs_partition[0], communicator);
    LA::MPI::Vector schur_rhs(locally_owned_dofs_partition[1], communicator);

    // First the pressure, using the pressure Schur complement of the first
    // two equations:
    {
      m_inverse.vmult(tmp, system_rhs.block(0));
      system_matrix.block(1, 0).vmult(schur_rhs, tmp);
      schur_rhs -= system_rhs.block(1);

      SchurComplement schur_complement(system_matrix, m_inverse,
                                       locally_owned_dofs_partition[0], communicator);

      ApproximateSchurComplement approximate_schur_complement(system_matrix,
                                                              locally_owned_dofs_partition[0], communicator);

      InverseMatrix<ApproximateSchurComplement> preconditioner(approximate_schur_complement);

      SolverControl solver_control(effective_solution.block(1).size(),
                                   1e-12 * schur_rhs.l2_norm());
      SolverCG<LA::MPI::Vector>  cg(solver_control);
      cg.solve(schur_complement, effective_solution.block(1), schur_rhs, preconditioner);

      pcout << "   " << solver_control.last_step()
            << " CG Schur complement iterations for pressure." << std::endl;
    }

    // Now the velocity:
    {
      system_matrix.block(0, 1).vmult(tmp, effective_solution.block(1));
      tmp *= -1;
      tmp += system_rhs.block(0);

      m_inverse.vmult(effective_solution.block(0), tmp);
    }
    // Finally, we have to take care of the saturation equation. The first
    // business we have here is to determine the time step using the formula
    // in the introduction. Knowing the shape of our domain and that we
    // created the mesh by regular subdivision of cells, we can compute the
    // diameter of each of our cells quite easily (in fact we use the linear
    // extensions in coordinate directions of the cells, not the
    // diameter). Note that we will learn a more general way to do this in
    // step-24, where we use the GridTools::minimal_cell_diameter function.
    //
    // The maximal velocity we compute using a helper function to compute the
    // maximal velocity defined below, and with all this we can evaluate our
    // new time step length:
    solution = effective_solution;
    time_step = std::pow(0.5, double(n_refinement_steps)) / get_maximal_velocity();

    // The next step is to assemble the right hand side, and then to pass
    // everything on for solution. At the end, we project back saturations
    // onto the physically reasonable range:
    assemble_rhs_S();
    {
      SolverControl solver_control(system_matrix.block(2, 2).m(),
                                   1e-8 * system_rhs.block(2).l2_norm());
      LA::SolverCG  cg(solver_control, LA::SolverCG::AdditionalData());
      LA::MPI::PreconditionAMG  amg;
      amg.initialize(system_matrix.block(2,2));
      cg.solve(system_matrix.block(2, 2), effective_solution.block(2),
               system_rhs.block(2), amg);

      project_back_saturation();

      pcout << "   " << solver_control.last_step()
            << " CG iterations for saturation." << std::endl;
    }
    solution = effective_solution;

    old_solution = solution;

    locally_relevant_solution = solution;

  }


  // @sect4{TwoPhaseFlowProblem::output_results}

  // There is nothing surprising here. Since the program will do a lot of time
  // steps, we create an output file only every fifth time step and skip all
  // other time steps at the top of the file already.
  //
  // When creating file names for output close to the bottom of the function,
  // we convert the number of the time step to a string representation that
  // is padded by leading zeros to four digits. We do this because this way
  // all output file names have the same length, and consequently sort well
  // when creating a directory listing.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::output_results() const {
    if(timestep_number % 5 != 0)
      return;

    std::vector<std::string> solution_names;
    switch(dim) {
        case 2:
          solution_names = {"u", "v", "p", "S"};
          break;

        case 3:
          solution_names = {"u", "v", "w", "p", "S"};
          break;

        default:
          Assert(false, ExcNotImplemented());
    }

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, solution_names);

    data_out.build_patches(degree + 1);

    std::string output_name = "solution-" + Utilities::int_to_string(Utilities::MPI::n_mpi_processes(communicator)) + "proc-" +
                              Utilities::int_to_string(timestep_number, 4) + ".vtu"  ;
    data_out.write_vtu_in_parallel(output_name, communicator);
  }



  // @sect4{TwoPhaseFlowProblem::project_back_saturation}

  // In this function, we simply run over all saturation degrees of freedom
  // and make sure that if they should have left the physically reasonable
  // range, that they be reset to the interval $[0,1]$. To do this, we only
  // have to loop over all saturation components of the solution vector; these
  // are stored in the block 2 (block 0 are the velocities, block 1 are the
  // pressures).
  //
  // It may be instructive to note that this function almost never triggers
  // when the time step is chosen as mentioned in the introduction. However,
  // if we choose the timestep only slightly larger, we get plenty of values
  // outside the proper range. Strictly speaking, the function is therefore
  // unnecessary if we choose the time step small enough. In a sense, the
  // function is therefore only a safety device to avoid situations where our
  // entire solution becomes unphysical because individual degrees of freedom
  // have become unphysical a few time steps earlier.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::project_back_saturation() {
    for(unsigned int i = effective_solution.block(2).local_range().first;
                     i < effective_solution.block(2).local_range().second; ++i)
      if(effective_solution.block(2)(i) < 0)
        effective_solution.block(2)(i) = 0;
      else if(effective_solution.block(2)(i) > 1)
        effective_solution.block(2)(i) = 1;
  }


  // @sect4{TwoPhaseFlowProblem::get_maximal_velocity}

  // The following function is used in determining the maximal allowable time
  // step. What it does is to loop over all quadrature points in the domain
  // and find what the maximal magnitude of the velocity is.
  template<int dim>
  double TwoPhaseFlowProblem<dim>::get_maximal_velocity() const {
    QGauss<dim>        quadrature_formula(degree + 2);
    const unsigned int n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(fe, quadrature_formula, update_values);
    std::vector<Vector<double>> solution_values(n_q_points,
                                                Vector<double>(dim + 2));
    double                      max_local_velocity = 0;

    for(const auto& cell : dof_handler.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(solution, solution_values);

        for(unsigned int q = 0; q < n_q_points; ++q) {
            Tensor<1, dim> velocity;
            for(unsigned int i = 0; i < dim; ++i)
              velocity[i] = solution_values[q](i);

            max_local_velocity = std::max(max_local_velocity, velocity.norm());
        }
      }
    }

    return Utilities::MPI::max(max_local_velocity, MPI_COMM_WORLD);
  }


  // @sect4{TwoPhaseFlowProblem::run}

  // This is the final function of our main class. Its brevity speaks for
  // itself. There are only two points worth noting: First, the function
  // projects the initial values onto the finite element space at the
  // beginning; the VectorTools::project function doing this requires an
  // argument indicating the hanging node constraints. We have none in this
  // program (we compute on a uniformly refined mesh), but the function
  // requires the argument anyway, of course. So we have to create a
  // constraint object. In its original state, constraint objects are
  // unsorted, and have to be sorted (using the AffineConstraints::close
  // function) before they can be used. This is what we do here, and which is
  // why we can't simply call the VectorTools::project function with an
  // anonymous temporary object <code>AffineConstraints<double>()</code> as the
  // second argument.
  //
  // The second point worth mentioning is that we only compute the length of
  // the present time step in the middle of solving the linear system
  // corresponding to each time step. We can therefore output the present end
  // time of a time step only at the end of the time step.
  //
  // The function as it is here does actually not compute the results
  // found on the web page. The reason is, that even on a decent
  // computer it runs more than a day. If you want to reproduce these
  // results, set the final time at the end of the do loop to 250.
  template<int dim>
  void TwoPhaseFlowProblem<dim>::run() {
    make_grid();

    setup_dofs();

    setup_dofs_initial_condition();

    assemble_initial_condition();

    solve_initial_condition();

    double time = 0;

    do {
      pcout << "Timestep " << timestep_number << std::endl;

      assemble_system();

      solve();

      TimerOutput::Scope t(time_table, "Output results");
      output_results();

      time += time_step;
      ++timestep_number;
      pcout << "   Now at t=" << time << ", dt=" << time_step << '.'
            << std::endl
            << std::endl;
    } while (time <= end_time);
  }
} // namespace Step21


// @sect3{The <code>main</code> function}

// That's it. In the main function, we pass the degree of the finite element
// space to the constructor of the TwoPhaseFlowProblem object.  Here, we use
// zero-th degree elements, i.e. $RT_0\times DQ_0 \times DQ_0$. The rest is as
// in all the other programs.
int main(int argc, char** argv) {
  try {
      using namespace dealii;
      using namespace Step21;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

      std::string parameter_filename;
      if(argc >= 2)
        parameter_filename = argv[1];
      else
        parameter_filename = "step-21.cfg";

      const int dim = 2;
      TwoPhaseFlowProblem<dim> two_phase_flow_problem(parameter_filename);
      two_phase_flow_problem.run();
  }
  catch(std::exception &exc) {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
  }
  catch (...) {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }

  return 0;
}
