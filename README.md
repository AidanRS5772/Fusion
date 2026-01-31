<h2>The Merced Fusion Project: In Progress</h2>

This code base hosts the main code for all computational aspects of The Merced Fusion Project. The code base is split into two main parts: modeling of Vasov systems related to the [Internal Electrostatic Confinement](https://en.wikipedia.org/wiki/Inertial_electrostatic_confinement) Fusion devices, and Computational Experiments done for optimizing geometries of the IEC fusion devices.

<h4>Cathode Design</h4>

The main goal of the cathode design directory is to have computational evidence for the optimal geometry of the cathode in an IEC fusion device. The code is split into 4 main parts: geometry search, geometry generation, simulation, and statistical analysis.

<h6>Geometry Search</h6>

From a review of the literature, two key important features of an optimal cathode geometry are spherical symmetry combined with antipodal symmetry of the plasma facing cathod component. This combination allows for maximum stabalizations of the plasma and maximum transperency of the cathode to the plasma. The problem is actually generating a possible search space of geometries that abide these criteria. 
The solution used is to use a gradient descent algorithm over over N nodes in [Real Projective Space](https://en.wikipedia.org/wiki/Real_projective_space) of two dimensions with a culomb potential. This algorithm maximizes spherical symetry for 2N apratures while enforcing antipodal symetry because of antipodal identification in RP2.
We can then form a Delauney triangulation of the solution embeded on the two sphere and take the cograph of this triangulation to form a geometry after pruning the coplaner  cycles from the geometry.
Doing this over hundreds of apprature counts leads to plausable geometries for any number of apprature counts the would want to be considered. With familar geometries arising such as some of the platonic solids, many of the [Goldberg Polyhedra](https://en.wikipedia.org/wiki/Goldberg_polyhedron), and many others.

<h6>Geometry Generation</h6>

Due to the canidate geomtries having complex topologies of genus streatching into the hundreds and the nessacity of having the meshes of these geomtries form a closed two dimensional surface, generating the meshes for these geometries in an automatic fashion is non-trivial. This automatic generation was acomplished using the [Gmsh](https://gmsh.info/) C++ API as base point and creating abstraction classes on top that did automatic tracking of orientations and boundaries of curves and surfaces. With this creating a full volumetric render of the geometry with adjustable mesh resolution was achieved.

<h6>Monte Carlo Simulation</h6>

For the purpose of being able to identify the most optimal of the candidate geometries, numerically solving the [Poisson](https://en.wikipedia.org/wiki/Poisson's_equation) BVP over the the volumetric meshes is nessacary. For this the [deal.ii](https://dealii.org/) FEM C++ library is used.
Then a [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) simulation takes 10,000 samples the phase space of inital conditions for an arbitrary particle in the reactor measuring the likely hood of particle undergoing a fusion process. This allows the assesment of various geometries for the fusion capabilities in the ultra-high vaccum limit. The simulation samples a [Maxwell-Boltzman](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution) distribution at room temprature, tracking particle paths through the geometry detecting colisions with the geomtry surfaces present. The paths are found using symplectic integrators evaluating the Poisson potential and gradient using the interpolating evaluation deal.ii C++ API. The symplectic integrator is suplimented with a gradient descent phase space adjuster that is used to supliment long time integrations.

<h6>Statistical Analyis</h6>

The data of the monte carlo simulation is synthesized into a single distribution of the likely hood of particle to under go fusion. This synthesis is found through analytic means of integrating over a [Maxwell-Juttner](https://en.wikipedia.org/wiki/Maxwell%E2%80%93J%C3%BCttner_distribution) distrbution to the reactivity of a given particle passing through the confined plasma based on its energy times the number of pass throughs. These distributions are then fit variuos [Pareto](https://en.wikipedia.org/wiki/Pareto_distribution) adjacent distributions to form a compleate statistical pictute of the behavior of a particular geometry.

<h4>Vlasov Equations: In Progress</h4>

Solving a specific variant of the [Vlasov](https://en.wikipedia.org/wiki/Vlasov_equation) equations with a central potential found in IEC fusion reactors, for the purpose of numerically finding the optimal pressure regimes to fasilitate fusion reactions. Voltage and size estimates can be found via analytic means however geometry and pressure estimations are less amenable to direct analytical aproaches. Finding the optimal pressure range to maximize fusion events amounts to solving the single species Vlasov equations with a driving central potential with a piecwise cutoff. In the aproach here the long time, maximum entropy, limit is taken along with spherical symetry assumed. These simplifications make the equations amount to solving the spherical [Bratu](https://en.wikipedia.org/wiki/Liouville%E2%80%93Bratu%E2%80%93Gelfand_equation) equation in three dimensions which must be done numerically. Analysis is still on going.

