#  bitpit for levelset RBF generation
This file explains how to use the script linking the Levelset and RBF capabilities of bitpit to train an RBF collection that represents the signed distance field of STL-defined objects. 

## Dependency:
Eigen is used for its sparse system solver. It can be installed with `sudo apt install libeigen3-dev`.

## Generator script usage:
The `/examples/levelsetRBF_prototype` executable is the one to use to generate RBFs from `.stl` files. The syntax to use it is `./levelsetRBF_prototype <name_of_stl> <path_to_stl_file> <parameters_file>.param <error_calculation_type>`.

1. <name_of_stl> is the name of the STL file. The name must not include the `.stl` extension.
2. <path_to_stl_file> is the path, absolute or relative, that contains the desired STL file.
3. <parameters_file>.param contains as many columns as parameters that must be specified. The first line contains the name of the parameters, and the second line contains the numerical values. All numbers and words on a same line must be separated by ` `. The parameters are `nb_subdivision`, `nb_adapations`, `radius_ratio`, `base_function`, `mesh_range`, `max_num_threads`, `tolerance`. The parameters have default values, so they don't all have to be in the parameter file. 
	`nb_subdivision` dictates the background grid initial refinement. Having a higher number of `nb_subdivisions` leads to a more precise but more costly RBF.
	`nb_adaptations` dictates the number of adaptive refinement cycles. 
	The RBF will have a higher concentration of nodes around high error zones, which can decrease the training time, the evaluation time and the file size. 
	`radius_ratio` dictates the radius that each node has an influence over. This values should be kept low enough so that sharp geometry features are not lost.
	`base_function` dictates the base function type with an integer. The value `2` corresponds to a linear function and produces the lowest error in many cases.
	`mesh_range` dictates how far outside of the STL bounding box the background mesh should extend. Often, a `0.1` value, which represents a 10% increase on each side, is a good choice.
	`max_num_threads` dictates the number of threads that OMP can use. This reduced the training time by using multiple processes in parallel.
	`tolerance` dictates the expected tolerance when solving the system with all the nodes. 
4. <error_calculation_type> represents the type of calculation to be carried out for the representation error of the geometry. It can be `sphere`, `cube`, `funky`, `stl` or `none`. The three first alternatives are used for specific shapes that are hardcoded, for debugging purposes. The `stl` type computes the difference between the STL file and the RBF representation, but can be costly. The `none` type doesn't compute the error; it should be used when using the script only to generation an RBF representation.

An example of line that can be used is: `./levelsetRBF_protoype formefunky_elements_0016064 ./data/cubes/ levelsetRBF_prototype.param none`.

The script outputs a VTU file to visualize the obtained RBF and a `.input` file that has all the information relevant to the trained RBF.
