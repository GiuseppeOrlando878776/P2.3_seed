# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2

# Include any dependencies generated for this target.
include CMakeFiles/step-2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/step-2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/step-2.dir/flags.make

CMakeFiles/step-2.dir/step-2.cc.o: CMakeFiles/step-2.dir/flags.make
CMakeFiles/step-2.dir/step-2.cc.o: step-2.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/step-2.dir/step-2.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/step-2.dir/step-2.cc.o -c /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2/step-2.cc

CMakeFiles/step-2.dir/step-2.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/step-2.dir/step-2.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2/step-2.cc > CMakeFiles/step-2.dir/step-2.cc.i

CMakeFiles/step-2.dir/step-2.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/step-2.dir/step-2.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2/step-2.cc -o CMakeFiles/step-2.dir/step-2.cc.s

# Object files for target step-2
step__2_OBJECTS = \
"CMakeFiles/step-2.dir/step-2.cc.o"

# External object files for target step-2
step__2_EXTERNAL_OBJECTS =

step-2: CMakeFiles/step-2.dir/step-2.cc.o
step-2: CMakeFiles/step-2.dir/build.make
step-2: /home/giuseppe/Scaricati/dealii-9.1.1/build/lib/libdeal_II.g.so.9.1.1
step-2: /home/giuseppe/bin/p4est/DEBUG/lib/libp4est.so
step-2: /home/giuseppe/bin/p4est/DEBUG/lib/libsc.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_system.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_thread.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_regex.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
step-2: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_pike-blackbox.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_trilinoscouplings.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_piro.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_rol.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stokhos_muelu.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stokhos_ifpack2.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stokhos_amesos2.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stokhos_tpetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stokhos_sacado.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stokhos.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_rythmos.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_muelu-adapters.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_muelu-interface.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_muelu.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_moertel.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_locathyra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_locaepetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_localapack.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_loca.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_noxepetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_noxlapack.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_nox.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_phalanx.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_intrepid.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teko.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stratimikos.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stratimikosbelos.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stratimikosaztecoo.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stratimikosamesos.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stratimikosml.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_stratimikosifpack.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_ifpack2-adapters.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_ifpack2.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_anasazitpetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_ModeLaplace.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_anasaziepetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_anasazi.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_komplex.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_amesos2.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_shylu.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_belostpetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_belosepetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_belos.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_ml.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_ifpack.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_zoltan2.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_pamgen_extras.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_pamgen.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_amesos.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_galeri-xpetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_galeri-epetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_aztecoo.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_dpliris.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_isorropia.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_optipack.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_xpetra-sup.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_xpetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_thyratpetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_thyraepetraext.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_thyraepetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_thyracore.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_epetraext.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_trilinosss.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_tpetraext.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_tpetrainout.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_tpetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_kokkostsqr.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_tpetraclassiclinalg.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_tpetraclassicnodeapi.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_tpetraclassic.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_triutils.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_globipack.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_shards.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_zoltan.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_epetra.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_sacado.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_rtop.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_kokkoskernels.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teuchoskokkoscomm.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teuchoskokkoscompat.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teuchosremainder.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teuchosnumerics.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teuchoscomm.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teuchosparameterlist.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_teuchoscore.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_kokkosalgorithms.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_kokkoscontainers.so
step-2: /usr/lib/x86_64-linux-gnu/libtrilinos_kokkoscore.so
step-2: /usr/lib/x86_64-linux-gnu/libsmumps.so
step-2: /usr/lib/x86_64-linux-gnu/libdmumps.so
step-2: /usr/lib/x86_64-linux-gnu/libcmumps.so
step-2: /usr/lib/x86_64-linux-gnu/libzmumps.so
step-2: /usr/lib/x86_64-linux-gnu/libpord.so
step-2: /usr/lib/x86_64-linux-gnu/libmumps_common.so
step-2: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so
step-2: /usr/lib/x86_64-linux-gnu/libtbb.so
step-2: /usr/lib/x86_64-linux-gnu/libz.so
step-2: /usr/lib/x86_64-linux-gnu/libptscotch.so
step-2: /usr/lib/x86_64-linux-gnu/libptscotcherr.so
step-2: /usr/lib/x86_64-linux-gnu/libscotch.so
step-2: /usr/lib/x86_64-linux-gnu/libscotcherr.so
step-2: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
step-2: /usr/lib/x86_64-linux-gnu/libumfpack.so
step-2: /usr/lib/x86_64-linux-gnu/libcholmod.so
step-2: /usr/lib/x86_64-linux-gnu/libccolamd.so
step-2: /usr/lib/x86_64-linux-gnu/libcolamd.so
step-2: /usr/lib/x86_64-linux-gnu/libcamd.so
step-2: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
step-2: /usr/lib/x86_64-linux-gnu/libamd.so
step-2: /usr/lib/x86_64-linux-gnu/libparpack.so
step-2: /usr/lib/x86_64-linux-gnu/libarpack.so
step-2: /usr/lib/x86_64-linux-gnu/libscalapack-openmpi.so
step-2: /usr/lib/x86_64-linux-gnu/liblapack.so
step-2: /usr/lib/x86_64-linux-gnu/libblas.so
step-2: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempif08.so
step-2: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_usempi_ignore_tkr.so
step-2: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_mpifh.so
step-2: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
step-2: /usr/lib/x86_64-linux-gnu/libslepc.so
step-2: /usr/lib/x86_64-linux-gnu/libpetsc.so
step-2: CMakeFiles/step-2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable step-2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/step-2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/step-2.dir/build: step-2

.PHONY : CMakeFiles/step-2.dir/build

CMakeFiles/step-2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/step-2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/step-2.dir/clean

CMakeFiles/step-2.dir/depend:
	cd /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2 /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2 /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2 /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2 /home/giuseppe/bin/P2.3_seed/exercises/lab02/step-2/CMakeFiles/step-2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/step-2.dir/depend

