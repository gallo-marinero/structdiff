import freud, sys, ase.io.vasp, os, ase
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from ase import Atoms
import numpy as np

#___READ STRUCTURES___
# Define path to the folder where all files are contained
directory='test/'
# Define the file that is going to be read (OUTCAR or POSTCAR)
# If POSTCAR, then read as: ase.io.vasp.read_vasp("/POSCAR")
car='/POSCAR'
# All structures are stored in the dict 'structures'
structures={}
# Get all the folders (structures) in 'directory', not sorted
liststr=os.listdir(directory)
for dir_structure in liststr:
# Loading the VASP converged structure (either POSCAR or OUTCAR)
    cell_dir=directory+dir_structure+car
    cell = ase.io.vasp.read_vasp(cell_dir)
    cell.set_pbc((True, True, True))
    positions=np.array(cell.get_positions())
# Get attributes in Object (to be able to call them)
    box = freud.box.Box.from_box(np.array(cell.get_cell()))
# 'structures' is a dictionary with the name of the system ('name') as key and ['box', 'positions'] as value:
    structures[str(dir_structure)]=[box,positions]
#___COMPUTE___ 
# Computes clusters of particles' local environments
def get_env_motifmatch(system, motif, threshold, num_neighs, registration):
    neighs = {'num_neighbors': num_neighs}
    match = freud.environment.EnvironmentMotifMatch()
    match.compute(system, motif, threshold, neighbors=neighs, registration=registration)
    return match.matches, match.point_environments
#___VARIABLE DEFINITION___ 
# Dict containing the matches
env_motifmatch_match={}
# Dict containing the environments
env_motifmatch_env={}
#___CALCULATION OF ENVIRONMENT MOTIF MATCH___
# Store the motif (given by name in motif_name) in variable motif
motif_name='XHNHHH'
motif=structures[motif_name][1]

for name, (box, positions) in structures.items():
# Store the system in system variable 
        system=(box,positions)
# Compute dissimilarty test from scipy.procrustes
        std_motif, std_str, disparity=procrustes(motif,positions)
# Compute Hausdorff distance from scipy.spatial
        print('Analyzing system:', name)
        print('Hausdorff distance',directed_hausdorff(motif,positions))
        print('Disparity', disparity)
        env_motifmatch_match[name], env_motifmatch_env[name] = get_env_motifmatch(system,
                motif,threshold=.05, num_neighs=100, registration=True)
        print('Environment Motif Match analysis against', motif_name)
        print(env_motifmatch_match[name],'\n')
