import freud, sys, ase.io.vasp, os, collections, csv, json, re, itertools, time
from progress.bar import IncrementalBar
from ase import Atoms
from sh import gunzip
from scipy.spatial import procrustes
from scipy.spatial.distance import directed_hausdorff
from ase.build import make_supercell
from ase.io import read
import numpy as np
import pyscal.core as pc
import statistics as st
from matplotlib import use
import matplotlib.pyplot as plt
#___READ STRUCTURES___
#
folder='set/'
# Go to the execution directory. A folder named 'set' with the structures must
# be present
os.chdir(os.getcwd())
# Define the file that is going to be read (OUTCAR or POSTCAR)
# If POSTCAR, then read as: ase.io.vasp.read_vasp("/POSCAR")
car='/POSCAR'
# Store systems in the dict 'structures'
structures={}
# All structures are stored in the dict 'str_pyscal' for pyscal analysis
str_pc={}
# Get all the folders (structures) in 'directory', unordered
liststr=os.listdir(folder)
# Order them
liststr.sort()
# List with the metal atoms
mlist=[]
print('\n-------------------------------------------------')
print('Evaluation of structural descriptors for crystals')
print('-------------------------------------------------\n')
bar=IncrementalBar('Loading structures', max=len(liststr))
for dir_structure in liststr:
    bar.next()
# Loading the VASP converged structure (either POSCAR or OUTCAR)
# The 'set' directory must exist where the code is executed
    cell_dir=folder+dir_structure+car
    # Check if CAR is gzipped and if so, unzip it
    if os.path.isfile(cell_dir+'.gz') and not os.path.isfile(cell_dir):
        gunzip(cell_dir+'.gz')
    cell = ase.io.vasp.read_vasp(cell_dir)
    cell.set_pbc((True, True, True))
# Delete Lithium atoms
    del cell[cell.numbers== 3]
# Instructions to build a supercell
#    p=[[2,0,0],[0,2,0],[0,0,2]]
#    cell=make_supercell(origcell,p)
# 'structures' is a dictionary with the name of the system ('name') as key and ['box', 'positions'] as value:
# box in structures[name][0]
# positions in structures[name][1]
    positions=np.array(cell.get_positions())
    index=[]
    at_num=[]
    symbols=np.array(cell.get_chemical_symbols())
# Split the name to get the metal, which is in position 2
    split=list(filter(None,re.split('(\d+)',dir_structure)))
# Append metal to the list with the metals
    mlist.append(split[2])
    mlist.append(split[4])
    for at in cell:
        index.append(at.index)
        at_num.append(at.number)
# Get attributes in Object (to be able to call them)
#        for att in dir(i):
#            print (att, getattr(i,att))
    box = freud.box.Box.from_box(np.array(cell.get_cell()))
# Set as filename the symbols extracted from the cell
#    structures[str(cell.symbols)]=[box,positions]
# Set as filename the name of the folder (from the DFT calculation)
    structures[str(dir_structure)]=[box,positions,index,at_num,symbols]
# Read structures for pyscal
    str_pc[str(dir_structure)]=pc.System()
    str_pc[str(dir_structure)].read_inputfile(cell,format='ase',customkeys='symbols')
bar.finish()
# Remove duplicates from the list of metals
mlist=list(dict.fromkeys(mlist))
v=False

# Print the number of atoms considered for each structure 
# Useful for checking whether Li atoms are or not discarded
for name, (box,positions, index, at_num, symbols) in structures.items():
    print(name, len(positions))
if v:
    print("These two vectors must coincide")
# Print the size of the box for cell as read by Freud
    print("The length vector: {}".format(box.L))
# Print the size of the box for cell as read from VASP converged file
    print("The length vector: {}".format(cell.cell.lengths()))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter3D(positions[:,0],positions[:,1],positions[:,2])
#plt.show()

#___COMPUTE DESCRIPTORS___
#
# Compute Voronoi neighbor list, stored in 'nlist', removing those with small Voronoi facets.
# Also compute the Steinhard order parameters 'descriptor' for l=values
# If descriptor='q' calculate Ql order parameters
# If descriptor='w' calculate Wl order parameters
def get_feat(box, positions, structure, values, average, weighted,
        wl_normalize, descriptor, what):
    voro = freud.locality.Voronoi()
    voro.compute(system=(box, positions))
    nlist = voro.nlist.copy()
    nlist.filter(nlist.weights > 0.1)
    features = {}
    order = {}
    var = {}
    wl=False
    if descriptor=='w':
        wl=True
    for l in values:
        featl = freud.order.Steinhardt(l=l, average=average, wl=wl,
                weighted=weighted, wl_normalize=wl_normalize)
        featl.compute(system=(box, positions), neighbors=nlist)
# Get the Steinhardt order parameters for each 'l'        
        features[descriptor+'{}'.format(l)] = getattr(featl,what)
# Get the global order of Steinhardt order parameters for each 'l'        
        order[descriptor+'{}'.format(l)] = featl.order
# Get the variance of Steinhardt order parameters for each 'l'        
# Due to a bug in statistics, the variance must be calculated from np.array and
# precission = 64 (https://bugs.python.org/issue39218)
        var[descriptor+'{}'.format(l)] = st.variance(np.array(features[descriptor+'{}'.format(l)], dtype=np.float64))
        #print(var[descriptor+'{}'.format(l)])
    return order, features, var

# Compute SOPs for the environment of the metal atoms
def get_feat_metal(box, positions, at_num, structure, values, average, weighted, descriptor, what):
    aq = freud.locality.AABBQuery(box,positions)
    nlist=aq.query(positions,{'mode':'nearest','num_neighbors':4}).toNeighborList()
    features = {}
    order = {}
    var = {}
    wl=False
    if descriptor=='w':
        wl=True
    for l in values:
        featl = freud.order.Steinhardt(l=l, average=average, wl=wl, weighted=weighted)
        featl.compute(system=(box, positions), neighbors=nlist)
# Get the Steinhardt order parameters for each 'l'        
        features[descriptor+'{}'.format(l)] = getattr(featl,what)
# Get the global order of Steinhardt order parameters for each 'l'        
        order[descriptor+'{}'.format(l)] = featl.order
# Get the variance of Steinhardt order parameters for each 'l'        
        var[descriptor+'{}'.format(l)] = st.variance(np.array(features[descriptor+'{}'.format(l)], dtype=np.float64))
    return order, features, var

# Computes clusters of particles' local environments
def get_env_motifmatch(system, motif, threshold, num_neighs, registration):
    neighs = {'mode':'nearest','num_neighbors': num_neighs}
    match = freud.environment.EnvironmentMotifMatch()
    match.compute(system, motif, threshold, neighbors=neighs, registration=registration)
    return match.matches, match.point_environments

# Compute Bond Order parameters
def get_bondorder(box, positions, structure, bins):
    bondorder = freud.environment.BondOrder(bins)
    bondorder_arr = bondorder.compute(system=(box, positions), neighbors={'num_neighbors': 10}).bond_order
    
    return bondorder_arr

#___INPUT DEFINITION___

#  PYSCAL DEFINITIONS
steinhardt_pc=True
# Define whether SOPs are averaged
#averaged=False
# Choose exponent for neighbour search
voroexp=2
# Quantum numbers for the SOPs 
pc_vals=[2,4,6,8,10,12]

# FREUD DEFINITIONS
bins=3
bondorder=False
steinhardt=True
env_motifmatch=False
motif_name='Li16H3He1X4Y16Y4'
# Specify whether Steinhardt parameters are going to be evaluated for metals
# only too 
tetra=False
# Create a prefix to indicate whether parameters are averaged or not
prefix=''
#if average:
#    prefix='averaged_'
#if weighted:
#    prefix +='weighted_'
# Dict that stores 'Ql' Steinhard parameters, as 'name':'ql', being l the number
structure_feat={}
structure_feat_av={}
structure_feat_wt={}
structure_feat_avwt={}
structure_feat_norm={}
# Dict that stores the variance of the 'Ql' Steinhard parameters, as 'name':'ql', being l the number
structure_var={}
structure_var_av={}
structure_var_wt={}
structure_var_avwt={}
structure_var_norm={}
# Dict containing the system wide normalization of the Ql/Wl order parameter
structure_order={}
structure_order_av={}
structure_order_wt={}
structure_order_avwt={}
structure_order_norm={}
# Dict containing the bond order parameter
structure_bondorder={}
# Dict containing the environment cluster indices
env_cluster_idx={}
# and environment
env_cluster_env={}
# Dict containing the matches
env_motifmatch_match={}
# Dict containing the environments
env_motifmatch_env={}

# Evaluate SOPs with pyscal
# Dict to store the SOPs
q_pc={}
qprint={}
q={}
qav={}
print('\n')
print('! ! ! Li atoms were removed')
print('\n')
bar= IncrementalBar('Calculating descriptors',max=len(liststr))
if steinhardt_pc:
  with open('data.dat', 'w') as f, open('glb_data.dat','w') as glbf:
# Create the headers so that pandas can appropiately read the file
    f.write('Structure,atom,voroCN,sannCN,addCN,VorVol,vAvVol,TetrAng')
    glbf.write('Structure')
    for i in range(9):
        f.write(',chipar'+str(i+1))
# Loops over SOP parameters: 
# Q and averaged Q parameters
    for i in pc_vals:
        f.write(',q_'+str(i)+',avq_'+str(i)+',wtq_'+str(i))
        glbf.write(',q_'+str(i)+',avq_'+str(i)+',wtq_'+str(i)+',avwtq_'+str(i))
        glbf.write(',varq_'+str(i)+',varavq_'+str(i)+',varwtq_'+str(i)+',varavwtq_'+str(i))
# W and averaged W parameters
    for i in pc_vals:
        f.write(',w_'+str(i)+',avw_'+str(i)+',wtw_'+str(i))
        glbf.write(',w_'+str(i)+',avw_'+str(i)+',wtw_'+str(i)+',avwtw_'+str(i))
        glbf.write(',normw_'+str(i))
        glbf.write(',varw_'+str(i)+',varavw_'+str(i)+',varwtw_'+str(i)+',varavwtw_'+str(i))
        glbf.write(',varwnorm_'+str(i))
    for i in pc_vals:
        glbf.write(',dis_'+str(i))
    f.write('\n')
    for name in structures:
        bar.next()
        glbf.write('\n'+name)
#        print('\n'+name)
#        print('PyMod', 'M', 'Vor', 'Sann', 'Adpt', 'VorVol', 'vAvVol',
#                'TetrAng', '       Chiparams        ', ' TetrAng',' vVector')

#########
# FREUD #
#########
# Calculate the system wide normalization of the 𝑞𝑙or 𝑤𝑙order parameter and
# SOPs variance
        box=structures[name][0]
        positions=structures[name][1]
# Calculate q:
        descriptor='q'
        wl_normalize=False
# q not averaged, not weighted
        average=False
        weighted=False
        structure_order[name], structure_feat[name], structure_var[name] = get_feat(box, positions, name,
                pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
# q averaged
        average=True
        weighted=False
        structure_order_av[name], structure_feat_av[name], structure_var_av[name] = get_feat(box, 
                positions, name, pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
# q weighted
        average=False
        weighted=True
        structure_order_wt[name], structure_feat_wt[name], structure_var_wt[name] = get_feat(box, 
                positions, name, pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
# q weighted and averaged
        average=True
        weighted=True
        structure_order_avwt[name], structure_feat_avwt[name], structure_var_avwt[name] = get_feat(box, 
                positions, name, pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
        for j in pc_vals:
            glbf.write(','+str(structure_order[name]['q'+str(j)])+','+str(structure_order_av[name]['q'+str(j)]))
            glbf.write(','+str(structure_order_wt[name]['q'+str(j)]))
            glbf.write(','+str(structure_order_avwt[name]['q'+str(j)]))
            glbf.write(','+str(structure_var[name]['q'+str(j)]))
            glbf.write(','+str(structure_var_av[name]['q'+str(j)]))
            glbf.write(','+str(structure_var_wt[name]['q'+str(j)]))
            glbf.write(','+str(structure_var_avwt[name]['q'+str(j)]))
# Calculate w:
        descriptor='w'
# w not averaged, not weighted
        average=False
        weighted=False
        structure_order[name], structure_feat[name], structure_var[name] = get_feat(box, positions, name,
                pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
# w averaged
        average=True
        weighted=False
        structure_order_av[name], structure_feat_av[name], structure_var_av[name] = get_feat(box,
                positions, name, pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
# w weighted
        average=False
        weighted=True
        structure_order_wt[name], structure_feat_wt[name], structure_var_wt[name] = get_feat(box,
                positions, name, pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
# w weighted and averaged
        average=True
        weighted=True
        structure_order_avwt[name], structure_feat_avwt[name], structure_var_avwt[name] = get_feat(box,
                positions, name, pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
# w weighted and averaged and normalized
        average=False
        weighted=False
        wl_normalize=True
        structure_order_norm[name], structure_feat_norm[name], structure_var_norm[name] = get_feat(box,
                positions, name, pc_vals, average, weighted, wl_normalize, descriptor, 'particle_order')
        for j in pc_vals:
            glbf.write(','+str(structure_order[name]['w'+str(j)])+','+str(structure_order_av[name]['w'+str(j)]))
            glbf.write(','+str(structure_order_wt[name]['w'+str(j)]))
            glbf.write(','+str(structure_order_avwt[name]['w'+str(j)]))
            glbf.write(','+str(structure_order_norm[name]['w'+str(j)]))
            glbf.write(','+str(structure_var[name]['w'+str(j)]))
            glbf.write(','+str(structure_var_av[name]['w'+str(j)]))
            glbf.write(','+str(structure_var_wt[name]['w'+str(j)]))
            glbf.write(','+str(structure_var_avwt[name]['w'+str(j)]))
            glbf.write(','+str(structure_var_norm[name]['w'+str(j)]))
        #orderlst = [ structure_order[name][descriptor+str(l)] for l in values ]
        #varlst = [ structure_var[name][descriptor+str(l)] for l in values ]
        voro = freud.locality.Voronoi()
        voro.compute(system=(structures[name][0], structures[name][1]))
        nlist = voro.nlist
# Filter neighbors with tiny weight
        nlist.filter(nlist.weights > 0.1)
# Get number of neighs, not the pairs        
        neighs_count=nlist.neighbor_counts
# Get Voronoi atomic volumes
        vols=voro.volumes

##########
# PYSCAL #
##########
# Calculation of Voronoi-derived information (store different than for other
# neighbour-finding methods)
        vorodat=str_pc[name]
        vorodat.find_neighbors(method='voronoi', voroexp=voroexp)
# not averaged
        vorodat.calculate_q(pc_vals, averaged=False)
        q_pc[name]=vorodat.get_qvals(pc_vals, averaged=False)
        q=q_pc[name]
        vorodat.calculate_vorovector()
# Automatically calculate the averaged SOPs too
        vorodat.calculate_q(pc_vals, averaged=True)
        qav=vorodat.get_qvals(pc_vals, averaged=True)
# Calculate the disorder of the SOPs (averaged)
        for i in pc_vals:
            vorodat.calculate_disorder(q=i)
# Store atom objects in a variable        
            voroatms = vorodat.atoms
            disorder = [atm.disorder for atm in voroatms]
            glbf.write(','+str(np.mean(disorder)))
#
# Calculation of non-Voronoi information
        sann=str_pc[name]
# A 2.1 threshold avoids failure for certain template structures (Li16...)        
        sann.find_neighbors(method='cutoff',cutoff='sann',threshold=2.1)
        sannatms = sann.atoms
        str_pc[name].find_neighbors(method='cutoff',cutoff='adaptive')
#        str_pc[name].find_neighbors(method='number',nmax=4)
        str_pc[name].calculate_chiparams()
        str_pc[name].calculate_angularcriteria()
## Block to calculate & print the Radial Distribution Function (RDF)
        rdf=str_pc[name].calculate_rdf()
        plt.figure(0)
        plt.plot(rdf[1],rdf[0])
        plt.xlabel('Distance')
        plt.title('RDF '+name)
        plt.savefig('rdf_'+name)
        plt.clf()
## Store atom objects in a variable        
        atms = str_pc[name].atoms
# Print in screen
# Get atomic tags
        tag=[atm.custom['species'] for atm in atms]
##        fig, ax = plt.subplots()
        ind=0
#        print('SOP', pc_vals[1],len(q[1]),q[1])
        for i in range(len(atms)):
## Only print data for metallic atoms
#            if tag[i] in mlist:
                ind+=1
# Break if more than 4 metals are printed (this avoids larger prints when Se is
# present as halogen and metal)
#                if ind == 5:
#                    break
                f.write('{:^15s},{:^2s},{:2},{:2},{:2},{:.4f},{:.4f},{:.4f}'.format(name,tag[i],
                    voroatms[i].coordination,sannatms[i].coordination,atms[i].coordination,
                    voroatms[i].volume, voroatms[i].avg_volume,atms[i].angular))
# Loop over chiparams
                for j in range(9):
                    f.write(',{:3}'.format(atms[i].chiparams[j]))
# Loop over the SOPs and print them                
                for j in range(len(pc_vals)):
                    f.write(','+str(q[j][i])+','+str(qav[j][i]))
#                print(tag[i], voroatms[i].coordination,
#                        sannatms[i].coordination, '', atms[i].coordination, "%.2f" % voroatms[i].volume,
#                        "%.2f" % voroatms[i].avg_volume, atms[i].chiparams,
#                        "%.3f" % atms[i].angular, voroatms[i].vorovector,q[0][i])
                f.write('\n')
  f.close()
  bar.finish()
  print('\nDone.\n')
##        f.write('\n')
'''        
# Block to print Vorovector figure
                prefx=('ax'+str(ind))
                plt.figure(1)
                if ind==1:
                    ax.bar(np.array(range(4))-0.15, voroatms[i].vorovector,
                            width=0.1, label=str(i)+str(tag[i]))
                elif ind==2:
                    ax.bar(np.array(range(4))-0.05, voroatms[i].vorovector,
                            width=0.1, label=str(i)+str(tag[i]))
                if ind==3:
                    ax.bar(np.array(range(4))+0.05, voroatms[i].vorovector,
                            width=0.1, label=str(i)+str(tag[i]))
                if ind==4:
                    ax.bar(np.array(range(4))+0.15, voroatms[i].vorovector,
                            width=0.1, label=str(i)+str(tag[i]))

        ax.set_xticks([1,2,3,4])
        ax.set_xlim(0.5, 4.25)
        ax.set_xticklabels(['$n_3$', '$n_4$', '$n_5$', '$n_6$'])
        ax.set_ylabel("Number of faces")
        ax.legend()
        fig.savefig('vorovector_'+name)
    plt.clf()
#    for i in pc_vals:
#        vor_f='voro_'+i+'.txt'
#        with open('.txt', 'w') as f:

# Explore all combinations (order does not matter) of the SOPs calculated as
# specified in pc_vals
for j in itertools.combinations(range(len(pc_vals)),2):
        for name in structures:
# Plot the SOPs 
            plt.scatter(q_pc[name][j[0]],q_pc[name][j[1]],label=name)
            plt.xlabel("$q_{%i}$" % (pc_vals[j[0]]), fontsize=10)
            plt.ylabel("$q_{%i}$" % (pc_vals[j[1]]), fontsize=10)
            plt.legend(fontsize=7)
# Save the plot        
            if averaged:
                plt.title('pc q'+str(pc_vals[j[0]])+'-'+str(pc_vals[j[1]])+' vorexp='+str(voroexp)+' avrg')
                plt.savefig('pc_q'+str(pc_vals[j[0]])+'-'+str(pc_vals[j[1]])+'_vorexp'+str(voroexp)+'_avrg')
            else:
                plt.title('pc q'+str(pc_vals[j[0]])+'-'+str(pc_vals[j[1]])+' vorexp='+str(voroexp))
                plt.savefig('pc_q'+str(pc_vals[j[0]])+'-'+str(pc_vals[j[1]])+'_vorexp'+str(voroexp))
        plt.clf()
"""
if env_motifmatch:
# Store the motif (given by name in motif_name) in variable motif
    motif=structures[motif_name]
    aq = freud.locality.AABBQuery(motif[0],motif[1])
# Positions of the atoms in the motif
    pos=motif[1]
# Variable to store the positions of the motif
    motifpos={}
# Define the number of neighbors to consider for motif match
    num_neighs=12
    print('\n--- Environment motif match ---\n')
    print('Selected motif structure is:')
    print(motif_name+'\n')
    print('Find the',num_neighs,'nearest neighbors of the following atomic positions (metals):')
    for i in range(len(motif[4])):
# Search for the metals and find 'num_neighs' nearest neighbors
        if motif[4][i] in mlist:
            print(motif[2][i],motif[4][i],motif[1][i])
# Finde num_neigh nearest neighbors            
            ats=aq.query(motif[1][i],{'mode':'nearest','num_neighbors':num_neighs}).toNeighborList()
# Find the vectors from target particle and its neighbors
            neigpos=(pos[ats.point_indices]-pos[ats.query_point_indices])
            neigpos=motif[0].wrap(neigpos)

# Store in a dictionary the atomic positions of the neighbors            
            motifpos[motif[4][i]]=neigpos
            print(np.array(neigpos))
# For each 'name', i.e.: for each VASP optimized structure, call the selected function
    motif=structures[motif_name][1]
for name, (box, positions,index,at_num,symbols) in structures.items():
    if steinhardt:
# get_features and calculate Voronoi neighbors and Ql
        if tetra:
            structure_order[name], structure_feat[name], structure_var[name] = get_feat_metal(box, positions, at_num, name, values, average, weighted, descriptor, 'particle_order')
        else:
            structure_order[name], structure_feat[name], structure_var[name] = get_feat(box, positions, name, values, average, weighted, descriptor, 'particle_order')
# Get Bond Orders 
    elif bondorder:
        structure_bondorder[name] = get_bondorder(box, positions, name, bins)
# Get Environment Cluster
    elif env_motifmatch:
# Store the system in system variable 
        system=freud.AABBQuery(box,positions)
# Compute dissimilarty test from scipy.procrustes
        std_motif, std_str, disparity=procrustes(motif,positions)
# Compute Hausdorff distance from scipy.spatial
        print('\nAnalyzing system:', name)
        print('Hausdorff distance',directed_hausdorff(motif,positions))
        print('Disparity', disparity)
        env_motifmatch_match[name], env_motifmatch_env[name] = get_env_motifmatch(system,
                motifpos['He'], threshold=.50, num_neighs=4, registration=True)
        print('Environment Motif Match analysis against', motif_name)
        print(env_motifmatch_match[name],env_motifmatch_env[name])

if bondorder:
    for name in structure_order.keys():
        print(name,structure_bondorder[name][0])

if steinhardt:
# Write to file:
# the system wide normalization of the ql/wl order parameters 
# the variance of the Steinhardt order params
    filename=prefix+descriptor
    orderfile=filename+'_order.txt'
    varfile=filename+'_var.txt'
    with open(orderfile, 'w') as f, open(varfile, 'w') as fv:
    # Write header with the names of the columns
        f.write('Structure\t')
        fv.write('Structure\t')
        for l in values:
            f.write(descriptor+str(l)+' ')
            fv.write(descriptor+str(l)+' ')
        f.write('\n')
        fv.write('\n')
    # Write the name of the structure and the calculated values
        for name in structure_order.keys():
            f.write(name+'\t')
            fv.write(name+'\t')
            orderlst = [ structure_order[name][descriptor+str(l)] for l in values ]
            varlst = [ structure_var[name][descriptor+str(l)] for l in values ]
            for i in range(len(values)):
                f.write(str(orderlst[i])+' ')
                fv.write(str(varlst[i])+' ')
            f.write('\n')
            fv.write('\n')
    f.close()
    fv.close()

    for l in values:
        fname=filename+'{}'.format(l)
        plt.figure(figsize=(5, 5), dpi=200)
        for name in structures.keys():
# Print variance too in the legend
            var=str(round(structure_var[name][descriptor+str(l)],5))
# Print global order parameter too in the legend
            order=str(round(structure_order[name][descriptor+str(l)],3))
            plt.hist(structure_feat[name][descriptor+str(l)], bins=20,
                    label=name+' '+order+' '+var, alpha=0.7)
#plt.hist(structure_feat[name][calc], range=(range_min, range_max), bins=80, label=name, alpha=0.7)
#plt.title(r'$q_{{{l}}}$'.format(l=l))
        plt.ylabel("Frequency", fontsize=14)
        plt.xlabel('$q_{%i}$' % (l), fontsize=14)
        plt.legend(fontsize='xx-small')
        plt.title(fname)
        plt.savefig(fname)
        plt.clf()
# If requested to calculate only Steinhardt OPs for the 4 nearest neighbors of
# the metal atoms
    if tetra:
        for l in values:
            fname=filename+'{}'.format(l)
# Variable to store the Steinhardt order parameters for the metal
# taking their 4 nearest neighbors, for plotting the results later
            tetr_datplot=[]
            tetr_mets=[]
            symb=[]
            with open(fname+'_tetrahedra.txt', 'w') as f:
                for name, (box, positions,index,at_num,symbols) in structures.items():
# Split the name to get the metal, which is in position 2
                    split=list(filter(None,re.split('(\d+)',name)))
# Features were calculated for every atom in the cell. Put them in list format
# for each structure
                    list_feat=structure_feat[name][descriptor+str(l)]
                    feat=[]
                    for i in range(len(symbols)):
# Print only features for the metal(s)
                        if symbols[i] == split[2]:
                            feat.append(list_feat[i])
                            symb.append(symbols[i])
                    f.write(name+' ')
                    f.write(split[2]+' ')
                    tetr_datplot.append(feat)
                    for i in range(len(feat)):
                        f.write(str(feat[i])+' ')
                    f.write('\n')
            f.close()
            tetr_datplot=np.array(tetr_datplot)
            for l in values:
                plt.figure(figsize=(3, 2), dpi=300)
                for i in range(len(tetr_datplot[0])):
                    plt.hist(tetr_datplot[:,i], bins=20, label=i, alpha=0.7)
                    plt.legend(fontsize='xx-small')
                    plt.title(fname+'_tetrahedra'+str(i))
                    plt.savefig(fname+'_tetrahedra'+str(i))
                    plt.clf()
            
'''
