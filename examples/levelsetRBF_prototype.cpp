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

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include <Eigen/Sparse>

#include <omp.h>

using namespace std;
using namespace bitpit;

double clamp(double expression, double high, double low){
    if (expression<low)
        return low;
    if (expression>high)
        return high;
    return expression;
}

double signed_distance_box(std::array<double,3> point, std::array<double,3> half_lengths)
{
    //Analytical value calculations
    std::array<double, 3> abs_p;
    for (unsigned int d = 0; d < 3; ++d) {
        abs_p[d] = std::abs(point[d]);
    }
    std::array<double, 3> q;
    q = abs_p - half_lengths;
    std::array<double, 3> max_q_0;
    for (unsigned int i = 0; i < 3; ++i) {
        max_q_0[i] = std::max(q[i], 0.);
    }
    double max_q = std::max(q[0], std::max(q[1], q[3 - 1]));
    return norm2(max_q_0) + std::min(max_q, 0.);
}

double signed_distance_cone(std::array<double,3> point, double tan_base_angle, double height)
{
    // For a cone, the parameters are tan(base angle) and height
    std::vector<double> p_xz;
    p_xz.push_back(point[0]);
    p_xz.push_back(point[2]);
    std::vector<double> w;
    w.push_back(norm2(p_xz));
    w.push_back(point[1]);
    std::vector<double> intermediate_q;
    intermediate_q.push_back(height * tan_base_angle);
    intermediate_q.push_back(-height);
    double dot_w_q = std::inner_product(std::begin(w), std::end(w),
                                        std::begin(intermediate_q), 0.0);
    double dot_q_q = std::inner_product(std::begin(intermediate_q), std::end(intermediate_q),
                                        std::begin(intermediate_q), 0.0);
    std::vector<double> a;
    a = w - intermediate_q * clamp(dot_w_q / dot_q_q, 1., 0.);
    std::vector<double> b_intermediate1;
    b_intermediate1.push_back(clamp(w[0] / intermediate_q[0], 1., 0.));
    b_intermediate1.push_back(1.);
    std::vector<double> b_intermediate2;
    b_intermediate2.push_back(intermediate_q[0] * b_intermediate1[0]);
    b_intermediate2.push_back(intermediate_q[1] * b_intermediate1[1]);

    std::vector<double> b;
    b        = w - b_intermediate2;
    double k = (intermediate_q[1] > 0.) ? 1 : ((intermediate_q[1] < 0.) ? -1. : 0.);
    double d = std::min(std::inner_product(std::begin(a), std::end(a),
                                           std::begin(a), 0.0),
                        std::inner_product(std::begin(b), std::end(b),
                                           std::begin(b), 0.0));
    double s = std::max(k * (w[0] * intermediate_q[1] - w[1] * intermediate_q[0]),
                        k * (w[1] - intermediate_q[1]));

    return sqrt(d) * ((s > 0.) ? 1. : ((s < 0.) ? -1. : 0.));
}

double signed_distance_sphere(std::array<double,3> point,
                              double radius)
{
    return norm2(point) - radius;
}

double signed_distance_torus(std::array<double,3> point,
                              double ring_radius,
                              double ring_thickness)
{
    std::array<double,2> p_xy;
    p_xy[0] = point[0];
    p_xy[1] = point[1];
    std::array<double,2> q;
    q[0] = norm2(p_xy) - ring_radius;
    q[1] = point[2];
    return norm2(q) - ring_thickness;
}

double signed_distance_funky(std::array<double,3> point)
{
    // Cone
    double tan_base_angle = 0.5;
    double height         = 1.0;
    std::array<double,3> centered_pt_cone;
    centered_pt_cone[0] = point[0] - 0.5;
    centered_pt_cone[1] = point[1] - 1.0 - height;
    centered_pt_cone[2] = point[2] - 0.0;

    // Box
    std::array<double, 3> half_lengths;
    for (unsigned int d = 0; d < 3; ++d) {
        half_lengths[d] = 0.5 ;
    }
    std::array<double,3> centered_pt_cube1;
    std::array<double,3> centered_pt_cube2;
    centered_pt_cube1[0] = point[0] - 0.0 - half_lengths[0];
    centered_pt_cube1[1] = point[1] - 0.0 - half_lengths[1];
    centered_pt_cube1[2] = point[2] - 0.0 - half_lengths[2];
    centered_pt_cube2[0] = point[0] - 0.5 - half_lengths[0];
    centered_pt_cube2[1] = point[1] - 0.5 - half_lengths[1];
    centered_pt_cube2[2] = point[2] - 0.5 - half_lengths[2];

    // Torus
    double ring_radius = 0.5;
    double ring_thickness = 0.2;
    std::array<double,3> centered_pt_torus;
    centered_pt_torus[0] = point[0] - 0.0;
    centered_pt_torus[1] = point[1] - 1.0;
    centered_pt_torus[2] = point[2] - 0.0;

    // Sphere
    double radius = 0.5;
    std::array<double,3> centered_pt_sphere;
    centered_pt_sphere[0] = point[0] - 1.0;
    centered_pt_sphere[1] = point[1] - 1.5;
    centered_pt_sphere[2] = point[2] - 1.0;

    std::array<double,5> levelset;
    levelset[0] = signed_distance_sphere(centered_pt_sphere,radius);
    levelset[1] = signed_distance_box(centered_pt_cube1,half_lengths);
    levelset[2] = signed_distance_box(centered_pt_cube2,half_lengths);
    levelset[3] = signed_distance_cone(centered_pt_cone,tan_base_angle,height);
    levelset[4] = signed_distance_torus(centered_pt_torus,ring_radius,ring_thickness);
    double distance = DBL_MAX;
    for (const double &lvlset : levelset)
    {
        distance = std::min(lvlset, distance);
    }
    return distance;
}

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

    // We add here the default values
    std::vector<std::string> names = {"nb_subdivision",
                                      "nb_adaptions",
                                      "radius_ratio",
                                      "base_function",
                                      "mesh_range",
                                      "max_num_threads",
                                      "tolerance",
                                      "scaling"};
    std::vector<double>      values = {16,
                                       0,
                                       1,
                                       2,
                                       0.2,
                                       1,
                                       1e-8,
                                       1.0};
    for (int i = 0; i < names.size(); i++)
    {
        if (map.find("f") == map.end())
            map[names[i]].push_back(values[i]);
    }
}

void run(std::string filename,
        std::string data_path,
        std::string parameter_file,
        std::string error_type) {
    constexpr int dimensions(3);

    // Parsing the parameters
    std::map<std::string, std::vector<double>> parameters;
    parse_parameters(parameters, parameter_file," ");
    int nb_subdivision      = static_cast<int>(parameters["nb_subdivision"][0]);
    int nb_adaptions        = static_cast<int>(parameters["nb_adaptions"][0]);
    double radius_ratio     =                  parameters["radius_ratio"][0];
    int base_function       = static_cast<int>(parameters["base_function"][0]);
    double mesh_range       =                  parameters["mesh_range"][0];
    int max_num_threads     = static_cast<int>(parameters["max_num_threads"][0]);
    double TOL              =                  parameters["tolerance"][0];
    double scaling          =                  parameters["scaling"][0];


    std::vector<std::string> timers_name;
    std::vector<double> timers_values;

    timers_name.push_back("load_geometry");
    double time_start = MPI_Wtime();

    //LEVELSET PART
    //Input geometry
#if BITPIT_ENABLE_MPI
    std::unique_ptr<bitpit::SurfUnstructured> STL0(new bitpit::SurfUnstructured(dimensions - 1, MPI_COMM_NULL));
#else
    std::unique_ptr<bitpit::SurfUnstructured> STL0( new bitpit::SurfUnstructured (dimensions - 1) );
#endif
    bitpit::log::cout() << " - Loading stl geometry" << std::endl;
    // Make sure that the STL format is in binary (not ASCII)
    try {
        STL0->importSTL(data_path + filename + ".stl", true);
    }
    catch (const std::bad_alloc) {
        STL0->importSTL(data_path + filename + ".stl", false);
    }
    STL0->deleteCoincidentVertices();
    STL0->initializeAdjacencies();
    STL0->getVTK().setName("levelset");
    std::array<double, dimensions> center{};
    STL0->scale(scaling,scaling,scaling,center);
    bitpit::log::cout() << "n. vertex: " << STL0->getVertexCount() << std::endl;
    bitpit::log::cout() << "n. simplex: " << STL0->getCellCount() << std::endl;
    // Create initial octree mesh for levelset
    bitpit::log::cout() << " - Setting mesh" << std::endl;
    std::array<double, dimensions> stlMin, stlMax, meshMin, meshMax, delta;
    double h(0.), dh, dh_RBF_nodes;
    STL0->getBoundingBox(stlMin, stlMax);
    delta = stlMax - stlMin;
    meshMin = stlMin - mesh_range * delta;
    meshMax = stlMax + mesh_range * delta;
    for (int i = 0; i < dimensions; ++i) {
        h = std::max(h, meshMax[i] - meshMin[i]);
    }
    dh = h / nb_subdivision;
#if BITPIT_ENABLE_MPI
    bitpit::VolOctree mesh(dimensions, meshMin, h, dh, MPI_COMM_WORLD);
#else
    bitpit::VolOctree mesh(dimensions, meshMin, h, dh);
#endif
    mesh.initializeAdjacencies();
    mesh.initializeInterfaces();
    mesh.update();
    mesh.getVTK().setName("RBF_levelset");
    mesh.setVTKWriteTarget(PatchKernel::WriteTarget::WRITE_TARGET_CELLS_INTERNAL);

    timers_values.push_back(MPI_Wtime() - time_start);
    timers_name.push_back("compute_levelset");
    time_start = MPI_Wtime();


    // Set levelset configuration
    bitpit::LevelSet levelset;
    levelset.setMesh(&mesh);
    int id0 = levelset.addObject(std::move(STL0), 0);
    const bitpit::LevelSetObject &object0 = levelset.getObject(id0);
    std::vector<int> ids;
    levelset.getObject(id0).enableVTKOutput(bitpit::LevelSetWriteField::VALUE);
    levelset.setPropagateSign(true);
    levelset.setSizeNarrowBand(sqrt(3.0) * h);
    // Compute the levelset
    levelset.compute(id0);
    // Write levelset information
    mesh.write();
    bitpit::log::cout() << "Computed levelset within the narrow band... " << std::endl;


    // Adaptative Refinement
    std::vector<bitpit::adaption::Info> adaptionData_levelset;
    for (int r = 0; r < nb_adaptions; ++r) {
        for (auto &cell: mesh.getCells()) {
            long cellId = cell.getId();
            if (std::abs(object0.getValue(cellId)) < mesh.evalCellSize(cellId))
                mesh.markCellForRefinement(cellId);
        }
        adaptionData_levelset = mesh.update(true);
        levelset.update(adaptionData_levelset);
        mesh.write();
    }
    unsigned long nP_total = mesh.getCellCount();


    timers_values.push_back(MPI_Wtime() - time_start);
    timers_name.push_back("add_nodes_to_RBF");
    time_start = MPI_Wtime();


    // RBF PART
    bitpit::RBFBasisFunction basisFunction;
    switch (base_function) {
        case 0:
            basisFunction = bitpit::RBFBasisFunction::CUSTOM;
            break;
        case 1:
            basisFunction = bitpit::RBFBasisFunction::WENDLANDC2;
            break;
        case 2:
            basisFunction = bitpit::RBFBasisFunction::LINEAR;
            break;
        case 3:
            basisFunction = bitpit::RBFBasisFunction::GAUSS90;
            break;
        case 4:
            basisFunction = bitpit::RBFBasisFunction::GAUSS95;
            break;
        case 5:
            basisFunction = bitpit::RBFBasisFunction::GAUSS99;
            break;
        case 6:
            basisFunction = bitpit::RBFBasisFunction::C1C0;
            break;
        case 7:
            basisFunction = bitpit::RBFBasisFunction::C2C0;
            break;
        case 8:
            basisFunction = bitpit::RBFBasisFunction::C0C1;
            break;
        case 9:
            basisFunction = bitpit::RBFBasisFunction::C1C1;
            break;
        case 10:
            basisFunction = bitpit::RBFBasisFunction::C2C1;
            break;
        case 11:
            basisFunction = bitpit::RBFBasisFunction::C0C2;
            break;
        case 12:
            basisFunction = bitpit::RBFBasisFunction::C1C2;
            break;
        case 13:
            basisFunction = bitpit::RBFBasisFunction::C2C2;
            break;
        default:
            basisFunction = bitpit::RBFBasisFunction::LINEAR;
            break;
    }
    std::vector<double> values;
    std::vector<double> weights;
    std::vector<double> radii;
    std::vector<std::array<double, dimensions>> nodes;
    values.resize(nP_total);
    weights.resize(nP_total);
    nodes.resize(nP_total);
    radii.resize(nP_total);

    bitpit::RBF RBFObject;
    RBFObject.setMode(bitpit::RBFMode::PARAM);
    RBFObject.setFunction(basisFunction);

    bitpit::log::cout() << "Adding nodes to the RBF" << std::endl;
    for (size_t it_RBF = 0; it_RBF < nP_total; it_RBF++) {
        nodes[it_RBF] = mesh.evalCellCentroid(it_RBF);
        values[it_RBF] = levelset.getObject(id0).getValue(it_RBF);
        RBFObject.addNode(nodes[it_RBF]);
        radii[it_RBF] = mesh.evalCellSize(it_RBF) * radius_ratio;
    }
    RBFObject.setSupportRadius(radii);

    timers_values.push_back(MPI_Wtime() - time_start);
    timers_name.push_back("fill_RHS");
    time_start = MPI_Wtime();

    // Training the RBF
    bitpit::log::cout() << "Training RBFObject" << std::endl;
    // Initializing the matrix and vector
    Eigen::VectorXd b(nP_total);
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(nP_total, nP_total);
    // Prepare the filling
    omp_set_num_threads(max_num_threads);
    // Compute the values for b
    bitpit::log::cout() << "Filling the RHS" << std::endl;
    for (int j = 0; j < nP_total; j++) {
        b[j] = values[j];
    }

    timers_values.push_back(MPI_Wtime() - time_start);
    timers_name.push_back("fill_matrix");
    time_start = MPI_Wtime();

    // Compute the values for A
    bitpit::log::cout() << "Filling the matrix" << std::endl;
    #pragma omp parallel
    {
        Eigen::SparseMatrix<double, Eigen::RowMajor> A_private(nP_total, nP_total);
        #pragma omp for
        for (int i = 0; i < nP_total; i++) {
            for (int j = 0; j < nP_total; j++) {
                double dist = RBFObject.calcDist(i, j) / radii[j];
                double v_ij = RBFObject.evalBasis(dist);
                if (abs(v_ij) > TOL) {
                    A_private.insert(i, j) = v_ij;
                }
            }
        }
        #pragma omp critical
        {
            A = A + A_private;
        }
    }

    bitpit::log::cout() << "100%" << std::endl;
    bitpit::log::cout() << "Compressing the matrix" << std::endl;
    A.makeCompressed();

    timers_values.push_back(MPI_Wtime() - time_start);
    timers_name.push_back("solve_system_with_eigenCG");
    time_start = MPI_Wtime();

    bitpit::log::cout() << "Solving the system with Eigen" << std::endl;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper> cg;
    cg.setTolerance(TOL);
    cg.compute(A);
    if (cg.info() != Eigen::Success) {
        // decomposition failed
        bitpit::log::cout() << "Decomposition failed" << std::endl;
    }
    Eigen::VectorXd x = cg.solve(b);
    if (cg.info() != Eigen::Success) {
        // solving failed
        bitpit::log::cout() << "Solving failed" << std::endl;
    }
    bitpit::log::cout() << "Solved Eigen system" << std::endl;

    timers_values.push_back(MPI_Wtime() - time_start);
    timers_name.push_back("add_weights_to_RBFObject");
    time_start = MPI_Wtime();

    bitpit::log::cout() << "Adding weights to RBFObject" << std::endl;
    for (int i = 0; i < nP_total; i++) {
        weights[i] = x[i];
    }
    RBFObject.addData(weights);
    bitpit::log::cout() << "Added weights to RBFObject" << std::endl;
    bitpit::log::cout() << "Finished RBF training" << std::endl;


    std::size_t nMaxCellVertices = mesh.getCell(0).getVertexIds().size();

    timers_values.push_back(MPI_Wtime()-time_start);
    timers_name.push_back("output_RBF_and_analytical");
    time_start = MPI_Wtime();

    // RBF Output
    bitpit::log::cout() << "Outputting" << std::endl;
    std::vector<double> display_values;
    display_values.resize(nP_total);
    #pragma omp parallel
    {
        std::vector<double> display_values_private;
        display_values_private.resize(nP_total);
        BITPIT_CREATE_WORKSPACE(vertexCoordinates, std::array<double BITPIT_COMMA dimensions>, nMaxCellVertices,
                                ReferenceElementInfo::MAX_ELEM_VERTICES)
        #pragma omp for
        for (int i = 0; i < nP_total; i++) {
            Cell cell = mesh.getCell(i);
            const unsigned long global_id = cell.getId();
            mesh.getCellVertexCoordinates(global_id, vertexCoordinates);
            std::array<double, dimensions> point = cell.evalCentroid(vertexCoordinates);
            std::vector<double> temp_disp = RBFObject.evalRBF(point);
            display_values_private[global_id] = temp_disp[0]; //Only the first field is used, since there is only one
        }
        #pragma omp critical
        {
            for (int i = 0; i < nP_total; i++) {
                display_values[i] += display_values_private[i];
            }
        }
    }
    mesh.getVTK().addData<double>("RBF", VTKFieldType::SCALAR, VTKLocation::CELL, display_values);
    mesh.write();

    timers_values.push_back(MPI_Wtime()-time_start);
    timers_name.push_back("compute_L2_error_and_output_error");
    time_start = MPI_Wtime();

    // L2 Error
    double l2_error = 0;
    double l1_error = 0;
    double linf_error = 0;
    nb_subdivision = 128;
    double fine_dh = h/nb_subdivision;
    if (error_type ==  "cube"
        || error_type == "sphere"
        || error_type == "funky")
    {
        std::array<double, dimensions> half_lengths;
        for (unsigned int d = 0; d < dimensions; ++d) {
            half_lengths[d] = 0.1 ;
        }
        #pragma omp parallel for reduction (+:l2_error,l1_error,linf_error)
        for (int x_it = 0;
             x_it < nb_subdivision;
             x_it++)
        {
            for (int y_it = 0;
                 y_it < nb_subdivision;
                 y_it++)
            {
                for (int z_it = 0;
                     z_it < nb_subdivision;
                     z_it++)
                {
                    double x = meshMin[0] + 0.5*fine_dh + x_it * fine_dh;
                    double y = meshMin[1] + 0.5*fine_dh + y_it * fine_dh;
                    double z = meshMin[2] + 0.5*fine_dh + z_it * fine_dh;
                    std::array<double,dimensions> point;
                    point[0] = x;
                    point[1] = y;
                    point[2] = z;
                    double analytical_sdf;
                    if (error_type ==  "cube")
                        analytical_sdf = signed_distance_box(point,half_lengths);
                    else if (error_type ==  "sphere")
                        analytical_sdf = signed_distance_sphere(point,0.5);
                    else if (error_type ==  "funky")
                        analytical_sdf = signed_distance_funky(point);

                    std::vector<double> temp_disp_error;
                    temp_disp_error = RBFObject.evalRBF(point);
                    double rbf_sdf = temp_disp_error[0]; //Only the first field is used, since there is only one
                    l2_error += (analytical_sdf - rbf_sdf) * (analytical_sdf - rbf_sdf); //* cell_volume;
                }
            }
        }

        // Error with analytical formulas
        std::vector<double> values_RBF;
        values_RBF.resize(nP_total);
        std::vector<double> values_analytical;
        std::vector<double> values_abs_error;
        values_analytical.resize(nP_total);
        values_abs_error.resize(nP_total);

        #pragma omp parallel
        {
            std::vector<double> values_analytical_private;
            std::vector<double> values_abs_error_private;
            values_analytical_private.resize(nP_total);
            values_abs_error_private.resize(nP_total);
            BITPIT_CREATE_WORKSPACE(vertexCoordinates_l2error, std::array<double BITPIT_COMMA dimensions>, nMaxCellVertices,
                                    ReferenceElementInfo::MAX_ELEM_VERTICES)
            #pragma omp for
            for (int i = 0; i < nP_total; i++) {
                Cell cell = mesh.getCell(i);
                const unsigned long cell_id = cell.getId();
                mesh.getCellVertexCoordinates(cell_id, vertexCoordinates_l2error);
                std::array<double, dimensions> point = cell.evalCentroid(vertexCoordinates_l2error);
                std::vector<double> temp_disp_output;
                temp_disp_output = RBFObject.evalRBF(point);
                values_RBF[cell_id] = temp_disp_output[0]; //Only the first field is used, since there is only one
                //Analytical value calculations
                if (error_type == "cube")
                    values_analytical_private[cell_id] = signed_distance_box(point, half_lengths);
                else if (error_type == "sphere")
                    values_analytical_private[cell_id] = signed_distance_sphere(point, 0.5);
                else if (error_type == "funky")
                    values_analytical_private[cell_id] = signed_distance_funky(point);
                //Cell volumes calculation
                values_abs_error_private[cell_id] = abs(values_RBF[cell_id] - values_analytical_private[cell_id]);
            }
            #pragma omp critical
            {
                for (int i = 0; i < nP_total; i++) {
                    values_analytical[i] += values_analytical_private[i];
                    values_abs_error[i] += values_abs_error_private[i];
                }
            }
        }
        mesh.getVTK().addData<double>("abs_error", VTKFieldType::SCALAR, VTKLocation::CELL, values_abs_error);
        mesh.getVTK().addData<double>("analytical", VTKFieldType::SCALAR, VTKLocation::CELL, values_analytical);
        mesh.write();
    }
    else if (error_type == "none")
    {}
    else if (error_type == "stl")
    {
        // Calculation of l2 error in relation to a  finer mesh with the STL object
        nb_subdivision = 128.;
        bitpit::VolOctree mesh_fine_for_error(dimensions, meshMin, h, h/nb_subdivision, MPI_COMM_WORLD);
        mesh_fine_for_error.initializeAdjacencies();
        mesh_fine_for_error.initializeInterfaces();
        mesh_fine_for_error.update();
        std::unique_ptr<bitpit::SurfUnstructured> STL0_bis(new bitpit::SurfUnstructured(dimensions - 1, MPI_COMM_NULL));
        bitpit::log::cout() << " - Loading stl geometry" << std::endl;
        // Make sure that the STL format is in binary (not ASCII)
        try {
            STL0_bis->importSTL(data_path + filename + ".stl", true);
        }
        catch (const std::bad_alloc){
            STL0_bis->importSTL(data_path + filename + ".stl", false);
        }
        STL0_bis->deleteCoincidentVertices();
        STL0_bis->initializeAdjacencies();
        bitpit::LevelSet levelset_for_error;
        levelset_for_error.setMesh(&mesh_fine_for_error);
        levelset_for_error.setSizeNarrowBand(sqrt(3)*h);
        id0 = levelset_for_error.addObject(std::move(STL0_bis), 0);
        levelset_for_error.getObject(id0).enableVTKOutput(bitpit::LevelSetWriteField::VALUE);
        levelset_for_error.setPropagateSign(true);
        // Compute the levelset
        levelset_for_error.compute(id0);
        const PatchKernel::CellConstRange &cellWriteRange_l2error_STL = mesh_fine_for_error.getVTKCellWriteRange();
        #pragma omp parallel
        {
            BITPIT_CREATE_WORKSPACE(vertexCoordinates_l2error, std::array<double BITPIT_COMMA dimensions>, nMaxCellVertices,
                                    ReferenceElementInfo::MAX_ELEM_VERTICES)
            #pragma omp for reduction (+:l2_error,l1_error,linf_error)
            for (int i = 0; i < cellWriteRange_l2error_STL.evalSize(); i++) {
                Cell cell = mesh_fine_for_error.getCell(i);
                const unsigned long cell_id = cell.getId();
                mesh_fine_for_error.getCellVertexCoordinates(cell_id, vertexCoordinates_l2error);
                std::array<double, dimensions> point = cell.evalCentroid(vertexCoordinates_l2error);
                std::vector<double> temp_disp = RBFObject.evalRBF(point);
                //Analytical value calculations
                l2_error += (levelset_for_error.getObject(id0).getValue(cell_id) - temp_disp[0]) *
                            (levelset_for_error.getObject(id0).getValue(cell_id) - temp_disp[0]);
            }
        }
    }
    l2_error = sqrt(l2_error)/pow(nb_subdivision,dimensions);
    bitpit::log::cout() << "L2-error: "<<l2_error<< std::endl;

    timers_values.push_back(MPI_Wtime()-time_start);
    timers_name.push_back("output_RBF_to_txt");
    time_start = MPI_Wtime();

    //Outputting the combined RBF to a txt file
    ofstream fw("RBF_" + filename + ".input", std::ofstream::ate);
    if (fw.is_open()) {
        fw << "support_radius basis_function node_x node_y ";
        if constexpr (dimensions == 3)
        {
            fw << "node_z ";
        }
        fw << "weight \n";
        for (int line = 0; line < nP_total; line++) {
            // Set precision
            int PRECISION = 20; //number of decimals
            std::ostringstream streamObj_support_radius;
            std::ostringstream streamObj_x;
            std::ostringstream streamObj_y;
            std::ostringstream streamObj_z;
            std::ostringstream streamObj_weight;
            streamObj_support_radius << std::fixed;
            streamObj_x              << std::fixed;
            streamObj_y              << std::fixed;
            streamObj_z              << std::fixed;
            streamObj_weight         << std::fixed;
            streamObj_support_radius << std::setprecision(PRECISION);
            streamObj_x              << std::setprecision(PRECISION);
            streamObj_y              << std::setprecision(PRECISION);
            streamObj_z              << std::setprecision(PRECISION);
            streamObj_weight         << std::setprecision(PRECISION);
            streamObj_support_radius << radii[line];
            streamObj_x << nodes[line][0];
            streamObj_y << nodes[line][1];
            streamObj_z << nodes[line][2];
            streamObj_weight << weights[line];
            std::string strObj_support_radius = streamObj_support_radius.str();
            std::string strObj_x              = streamObj_x.str();
            std::string strObj_y              = streamObj_y.str();
            std::string strObj_z              = streamObj_z.str();
            std::string strObj_weight         = streamObj_weight.str();
            fw << strObj_support_radius << " " << base_function << " "<<
                  strObj_x << " " << strObj_y << " ";
            if constexpr (dimensions == 3)
            {
                fw << strObj_z << " ";
            }
            fw << strObj_weight << " ";
            if (line < nP_total -1)
                fw << "\n";
        }
        fw.close();
    } else bitpit::log::cout() << "Problem with opening file" << std::endl;
    bitpit::log::cout() << "Finished outputting" << std::endl;

    timers_values.push_back(MPI_Wtime()-time_start);

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
    std::string error_type     = argList[4];

    // Initialize the logger
	log::manager().initialize(log::MODE_COMBINE, true, nProcs, rank);
	log::cout() << log::fileVerbosity(log::INFO);
	log::cout() << log::disableConsole();

	// Run the example
    try {
        run(filename,
            data_path,
            parameter_file,
            error_type);
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
