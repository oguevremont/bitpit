#if BITPIT_ENABLE_MPI==1
#include <mpi.h>
#endif

#include "bitpit_surfunstructured.hpp"
#include "bitpit_voloctree.hpp"
#include "bitpit_levelset.hpp"
#include "bitpit_operators.hpp"
#include "bitpit_IO.hpp"
#include "bitpit_RBF.hpp"
#include "bitpit_common.hpp"
#include "bitpit_patchkernel.hpp"

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <unordered_map>
#include <vector>

#include <Eigen/Sparse>

#include <omp.h>

using namespace std;
using namespace bitpit;

void
parse_parameters(std::map<std::string, std::vector<double>> &map,
                 std::string                                 file,
                 const std::string                           delimiter)
{
    // fill a pair, first being a vector of vector name and the second being the
    // vector of vector associated with the vector name.
    std::ifstream myfile(file);
    // open the file.
    if (myfile.is_open())
    {
        std::string              line;
        std::vector<std::string> column_names;
        std::vector<double>      line_of_data;
        unsigned int             line_count = 0;

        while (std::getline(myfile, line))
        {
            // read the line and clean the resulting vector.
            std::vector<std::string> list_of_words_base;

            std::string s = line;
            size_t pos = 0;
            std::string token;
            while ((pos = s.find(delimiter)) != std::string::npos) {
                token = s.substr(0, pos);
                list_of_words_base.push_back(token);
                s.erase(0, pos + delimiter.length());
            }
            std::vector<std::string> list_of_words_clean;
            for (unsigned int i = 0; i < list_of_words_base.size(); ++i)
            {
                if (list_of_words_base[i] != "")
                {
                    list_of_words_clean.push_back(list_of_words_base[i]);
                }
            }
            // check if the line is contained words or numbers.
            if (line_count != 0)
            {
                line_of_data.resize(list_of_words_clean.size());
                for (int i = 0; i < line_of_data.size(); i++)
                {
                    line_of_data[i] = std::stod(list_of_words_clean[i]);
                }
                for (unsigned int i = 0; i < line_of_data.size(); ++i)
                {
                    map[column_names[i]].push_back(line_of_data[i]);
                }
            }
            else
            {
                // the line contains words, we assume these are the columns names.
                column_names = list_of_words_clean;
                for (unsigned int i = 0; i < list_of_words_clean.size(); ++i)
                {
                    std::vector<double> base_vector;
                    map[column_names[i]] = base_vector;
                }
            }
            ++line_count;
        }
        myfile.close();
    }
    else
        std::cout << "Unable to open file";

    ////// TODO Add mesh_min_x, mesh_min_y, mesh_min_z, mesh_max_x, mesh_max_y, mesh_max_z

    // We add here the default values
    std::vector<std::string> names = {"nb_initial_subdivision",
                                      "nb_uniform_adaptations",
                                      "scaling",
                                      "scaling_x",
                                      "scaling_y",
                                      "scaling_z",
                                      "dx",
                                      "dy",
                                      "dz",
                                      "mesh_min_x",
                                      "mesh_min_y",
                                      "mesh_min_z",
                                      "mesh_max_x",
                                      "mesh_max_y",
                                      "mesh_max_z",
                                      "swap_inside",
                                      "binary_stl"};

    std::vector<double> values = {16,   // nb_initial_subdivision
                                  0,    // nb_uniform_adaptations
                                  1.0,  // scaling
                                  1.0,  // scaling_x
                                  1.0,  // scaling_y
                                  1.0,  // scaling_z
                                  0.0,  // dx
                                  0.0,  // dy
                                  0.0,  // dz
                                  -0.6, // mesh_min_x
                                  -0.6, // mesh_min_y
                                  -0.6, // mesh_min_z
                                  +0.6, // mesh_max_x
                                  +0.6, // mesh_max_y
                                  +0.6, // mesh_max_z
                                  0,    // swap_inside
                                  0};   // binary_stl

    for (int i = 0; i < names.size(); i++) {
        if (map.find(names[i]) == map.end())
            map[names[i]].push_back(values[i]);
    }
}

void run(std::string filename,
        std::string data_path,
        std::string parameter_file)
{
    constexpr int dimensions(3);

    // Parsing the parameters
    std::map<std::string, std::vector<double>> parameters;
    parse_parameters(parameters, parameter_file, " ");
    int nb_subdivision        = static_cast<int>(parameters["nb_initial_subdivision"][0]);
    int nb_adaptions          = static_cast<int>(parameters["nb_uniform_adaptations"][0]);
    double scaling_global     = parameters["scaling"][0];
    double scaling_x          = parameters["scaling_x"][0];
    double scaling_y          = parameters["scaling_y"][0];
    double scaling_z          = parameters["scaling_z"][0];
    double dx                 = parameters["dx"][0];
    double dy                 = parameters["dy"][0];
    double dz                 = parameters["dz"][0];
    double mesh_min_x         = parameters["mesh_min_x"][0];
    double mesh_min_y         = parameters["mesh_min_y"][0];
    double mesh_min_z         = parameters["mesh_min_z"][0];
    double mesh_max_x         = parameters["mesh_max_x"][0];
    double mesh_max_y         = parameters["mesh_max_y"][0];
    double mesh_max_z         = parameters["mesh_max_z"][0];
    double swap_inside        = parameters["swap_inside"][0];
    double binary_stl         = parameters["binary_stl"][0];

    std::vector<std::string> timers_name;
    std::vector<double> timers_values;

    timers_name.push_back("load_geometry");
    double time_start = MPI_Wtime();

    //LEVELSET PART
    //Input geometry
#if BITPIT_ENABLE_MPI
    std::unique_ptr<bitpit::SurfUnstructured> STL0(new bitpit::SurfUnstructured(dimensions - 1, MPI_COMM_NULL));
#else
    std::unique_ptr<bitpit::SurfUnstructured> STL0(new bitpit::SurfUnstructured(dimensions - 1));
#endif
    bitpit::log::cout() << " - Loading stl geometry" << std::endl;
    // Make sure that the STL format is in the right format
    bool is_binary_stl = binary_stl > 0;
    try {
        STL0->importSTL(data_path + filename + ".stl", is_binary_stl);
    } catch (const std::bad_alloc) {
        STL0->importSTL(data_path + filename + ".stl", !is_binary_stl);
    }
    STL0->deleteCoincidentVertices();
    STL0->initializeAdjacencies();
    STL0->getVTK().setName("levelset");
    std::array<double, dimensions> center{};
    STL0->scale(scaling_global * scaling_x,
                scaling_global * scaling_y,
                scaling_global * scaling_z,
                center);
    bitpit::log::cout() << "n. vertex: " << STL0->getVertexCount() << std::endl;
    bitpit::log::cout() << "n. simplex: " << STL0->getCellCount() << std::endl;
    // Create initial octree mesh for levelset
    bitpit::log::cout() << " - Setting mesh" << std::endl;
    std::array<double, dimensions> stlMin, stlMax, meshMin, meshMax, delta;
    double h(0.), dh, dh_RBF_nodes;
    STL0->getBoundingBox(stlMin, stlMax);
    delta   = stlMax - stlMin;
    meshMin = stlMin;
    meshMax = stlMax;

    // Here we set the boundaries of the mesh depending on the parameters
    meshMin[0] = mesh_min_x;
    meshMin[1] = mesh_min_y;
    meshMin[2] = mesh_min_z;
    meshMax[0] = mesh_max_x;
    meshMax[1] = mesh_max_y;
    meshMax[2] = mesh_max_z;
    for (int i = 0; i < dimensions; ++i) {
        h = std::max(h, meshMax[i] - meshMin[i]);
    }
    dh = h / nb_subdivision;
#if BITPIT_ENABLE_MPI
    bitpit::VolOctree mesh(dimensions, meshMin, h, dh, MPI_COMM_WORLD);
    //bitpit::VolCartesian mesh(dimensions, meshMin, h, nb_subdivision);
#else
    bitpit::VolOctree mesh(dimensions, meshMin, h, dh);
#endif
    STL0->translate(dx, dy, dz);

    std::cout << "After rescale and translation" << std::endl;
    STL0->getBoundingBox(stlMin, stlMax);
    std::cout << "stlMin: " << stlMin[0] << " " << stlMin[1] << " " << stlMin[2] << std::endl;
    std::cout << "stlMax: " << stlMax[0] << " " << stlMax[1] << " " << stlMax[2] << std::endl;

    mesh.initializeAdjacencies();
    mesh.initializeInterfaces();
    mesh.update();
    mesh.getVTK().setName("voxelized_" + filename);
    mesh.setVTKWriteTarget(PatchKernel::WriteTarget::WRITE_TARGET_CELLS_INTERNAL);

    timers_values.push_back(MPI_Wtime() - time_start);
    timers_name.push_back("compute_levelset");
    time_start = MPI_Wtime();

    // Set levelset configuration
    bitpit::LevelSet levelset;
    levelset.setMesh(&mesh);
    int id0                               = levelset.addObject(std::move(STL0), 0);
    const bitpit::LevelSetObject &object0 = levelset.getObject(id0);
    std::vector<int> ids;
    levelset.getObject(id0).enableVTKOutput(bitpit::LevelSetWriteField::VALUE);
    levelset.setPropagateSign(true);
    levelset.setSizeNarrowBand(3.0 * h);
    // Compute the levelset
    levelset.compute(id0);
    // Write levelset information
    mesh.write();
    bitpit::log::cout() << "Computed levelset within the narrow band... " << std::endl;

    // Adaptative Refinement
    std::vector<bitpit::adaption::Info> adaptionData_levelset;
    for (int r = 0; r < nb_adaptions; ++r) {
        for (auto &cell : mesh.getCells()) {
            long cellId = cell.getId();
            if (std::abs(object0.getValue(cellId)) < mesh.evalCellSize(cellId))
                mesh.markCellForRefinement(cellId);
        }
        adaptionData_levelset = mesh.update(true);
        levelset.update(adaptionData_levelset);
        mesh.write();
    }
    unsigned long nP_total = mesh.getCellCount();

    // To feed the VTU voxelized SDF file to the neural network, we need a uniform cartesian grid.
    // This is handled by an external Python script

    timers_values.push_back(MPI_Wtime() - time_start);

    int nb_timers = timers_name.size();
    bitpit::log::cout()<< std::endl << "Timers" << std::endl;
    for (int t = 0; t < nb_timers; t++)
    {
        bitpit::log::cout() << timers_name.at(t) << ":" << timers_values.at(t) << std::endl;
    }
}

/*!
* Main program.
*/
int main(int argc, char *argv[])
{
    int nProcs = 1;
    int rank   = 0;

#if BITPIT_ENABLE_MPI==1
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (nProcs>1)
    {
        bitpit::log::cout() << "nProcs > 1 isn't supported" << std::endl;
        exit(1);
    }
#endif

    // Arguments
    std::vector<std::string> argList;
    for(int i=0;i<argc;i++)
        argList.emplace_back(argv[i]);
    std::string filename       = argList[1];
    std::string data_path      = argList[2];
    std::string parameter_file = argList[3];

    // Initialize the logger
	log::manager().initialize(log::MODE_COMBINE, true, nProcs, rank);
	log::cout() << log::fileVerbosity(log::INFO);
	log::cout() << log::disableConsole();

	// Run the example
    try {
        run(filename,
            data_path,
            parameter_file);
    }
    catch (const std::exception &exception) {
    log::cout() << exception.what();
    exit(1);
    }

#if BITPIT_ENABLE_MPI==1
	MPI_Finalize();
#endif
    return 0;
}
