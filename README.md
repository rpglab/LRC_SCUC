This set of codes implements our TPWRS paper "Security-Constrained Unit Commitment Considering Locational Frequency Stability in Low-Inertia Power Grids". 

The proposed locational-RoCoF-constrained security-constrained unit commitment (LRC-SCUC) model along with supporting methods are implemented, as well as two benchmark models. 


## Introduction:
* Two benchmark models: 
	* a traditional security-constrained unit commitment (T-SCUC).
	* an equivalent-RoCoF-constrained unit commitment (ERC-SCUC).
* The locational RoCoF constraints are derived from reduced IEEE 24-bus system dynamic models. Two RoCoF measuring windows (t1 = 0 and t2 = 0.5) are introduced to handle the oscillation in power system.
* Piecewise Linaerization Method (PWL) is used to linearize the original nonlinear locational RoCoF concstraints. 
* The test case includes time series profiles of renewable generation, electrical load.
* Multiple evaluation points combinations are set to reduce the PWL approximation errors.  

## Power system test data:
The test system used in this work is a modified IEEE 24-bus system: one area of the IEEE 73-bus system ("The IEEE Reliability Test System-1996. A report prepared by the Reliability Test System Task Force of the Application of Probability Methods Subcommittee" and link is <a class="" target="_blank" href="https://ieeexplore.ieee.org/document/780914">here</a>).
* 'dataFile24BusAllinertia41sen_T.data': test data including system configuration, genrator parameters, renewable generation profile, electrical load profile.
* 'dataFile24BusAllinertia41sen_4EP.data': test data including system configuration, genrator parameters, renewable generation profile, electrical load profile, PWL coefficient segments settings.

## PWL evaluation point data (./COE_Files):
* 'EP4_pwl_coefficients_busXX.xlsx': PWL coefficients of generator loss on bus XX in reduced power system model with RoCoF measuring time t=0.
* 'EP4_pwl_coefficients_busXX_t2.xlsx': PWL coefficients  of generator loss on bus XX in reduced power system model with RoCoF measuring time t=0.5.

## RoCoF constraints PWL coefficient data data:
* 'data_PWL_4EP_Type1.data': Medium generation range evaluation points settings.
* 'data_PWL_4EP_Type2.data': Large generation range evaluation points settings.
* 'data_PWL_4EP_Type3.data': Small generation range evaluation points settings.

## Python Codes to run security-constrainted unit commitment (SCUC)
* 'Sample_Codes_SCUC': A standard SCUC model.
* The evalution points data are provided by 'data_PWL_4EP_Type1.data', 'data_PWL_4EP_Type2.data', and 'data_PWL_4EP_Type3.data'.
* The load, renewable generation profiles are provided by 'dataFile24BusAllinertia41sen_T.dat', 'dataFile24BusAllinertia41sen_4EP.dat'.
* 'RoCoF_PWL_fitting.py': using PWL algorithm to linearize the RoCoF constraints, the output is optimized coefficient data.
* 'T_OPF_SCUC.py': define functions to build, solve, and save results for pyomo T-SCUC model.
* 'ERC_OPF_SCUC.py': define functions to build, solve, and save results for pyomo ERC-SCUC model.
* 'LRC_OPF_SCUC.py': define functions to build, solve, and save results for pyomo LRC-SCUC model.
* 'PWL_OPF_LRC_SCUC-Dual.py': define functions to build, solve, and save dual results for pyomo LRC-SCUC model.


## Python Codes Environment
* Recommand Python Version: Python 3.8
* Required packages: Numpy, pyomo
* Required a solver which can be called by the pyomo to solve the SCUC optimization problem.


## Steps to run SCUC simulation:
1. Set up the python environment.
2. Set the solver location: 'UC_function.py'=>'solve_UC' function=>UC_solver=SolverFactory('solver_name',executable='solver_location')
3. Run 'RoCoF_PWL_fitting.py': generate PWL coefficients for each generator bus.
4. Run SCUC models to get simulation results of test case.


## Simulation results profiles:
* 'T-SCUC_results' folder: T-SCUC simulation results, including each hour's generator dispatching data, generotor reserve data, generator status data.
* 'ERC-SCUC_results' folder: ERC-SCUC simulation results, including each hour's generator dispatching data, generotor reserve data, generator status data.   
* 'LRC-SCUC_results' folder: LRC-SCUC simulation results, including each hour's generator dispatching data, generotor reserve data, generator status data.



## Citation:
If you use any of our codes/data for your work, please cite the following paper as your reference:

Mingjian Tuo and Xingpeng Li, “Security-Constrained Unit Commitment Considering Locational Frequency Stability in Low-Inertia Power Grids”, *IEEE Transaction on Power Systems*, Oct. 2022.


(DOI: 10.1109/TPWRS.2022.3215915)

Paper website: https://rpglab.github.io/papers/MJ-Tuo_SCUC_LFS/


## Contributions:
Mingjian Tuo created this package. Xingpeng Li supervised this work.


## Contact:
Dr. Xingpeng Li

University of Houston

Email: xli83@central.uh.edu

Website: https://rpglab.github.io/


## License:
This work is licensed under the terms of the <a class="off" href="https://creativecommons.org/licenses/by/4.0/"  target="_blank">Creative Commons Attribution 4.0 (CC BY 4.0) license.</a>


## Disclaimer:
The author doesn’t make any warranty for the accuracy, completeness, or usefulness of any information disclosed; and the author assumes no liability or responsibility for any errors or omissions for the information (data/code/results etc) disclosed.
