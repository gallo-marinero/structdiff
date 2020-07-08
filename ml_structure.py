import freud, sys, ase.io.vasp, os, collections, csv, json, re
from ase import Atoms
from sh import gunzip
from ase.visualize import view
import numpy as np
from matplotlib import use
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#___READ STRUCTURES___
#
# Define path to the folder where all files are contained
directory='/home/gallo/work/struct_descriptors/d1/m5/test/'
# Define the file that is going to be read (OUTCAR or POSTCAR)
# If POSTCAR, then read as: ase.io.vasp.read_vasp("/POSCAR")
outcar='/CONTCAR'
# All structures are stored in the dict 'structures'
structures={}
for dir_structure in os.listdir(directory):
# Loading the VASP converged structure (either POSCAR or OUTCAR)
    cell_dir=directory+dir_structure+outcar
    # Check if OUTCAR is gzipped and if so, unzip it
    if os.path.isfile(cell_dir+'.gz'):
        gunzip(cell_dir+'.gz')
    cell = ase.io.vasp.read_vasp(cell_dir)
    cell.set_pbc((True, True, True))
# 'structures' is a dictionary with the name of the system ('name') as key and ['box', 'positions'] as value:
# box in structures[name][0]
# positions in structures[name][1]
# print(structures[name][1])
    positions=np.array(cell.get_positions())
    index=[]
    at_num=[]
    symbols=np.array(cell.get_chemical_symbols())
    for at in cell:
        index.append(at.index)
        at_num.append(at.number)
#        print(str(dir_structure),index,at_num)
# Get attributes in Object (to be able to call them)
#        for att in dir(i):
#            print (att, getattr(i,att))
    box = freud.box.Box.from_box(np.array(cell.get_cell()))
# Set as filename the symbols extracted from the cell
#    structures[str(cell.symbols)]=[box,positions]
# Set as filename the name of the folder (from the DFT calculation)
    structures[str(dir_structure)]=[box,positions,index,at_num,symbols]
    
v=False
# Some commands
#cell.get_cell()
#cell.get_chemical_symbols()
# Print the name of the structure
#print(cell.symbols)
# View the structure in rasmol
#view(cell, viewer='rasmol')

for name, (box,positions, index, at_num, symbols) in structures.items():
    print(name, len(positions))
if v:
    print("These two vectors must coincide")
# Print the size of the box for cell as read by Freud
    print("The length vector: {}".format(box.L))
# Print the size of the box for cell as read from VASP converged file
    print("The length vector: {}".format(cell.cell.lengths()))
    #print("Extended: {}".format(box))

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
def get_feat(box, positions, structure, values, average, weighted, descriptor, what):
    voro = freud.locality.Voronoi()
    voro.compute(system=(box, positions))
    nlist = voro.nlist.copy()
    nlist.filter(nlist.weights > 0.1)
    features = {}
    order = {}
    wl=False
    if descriptor=='w':
        wl=True
    for l in values:
        featl = freud.order.Steinhardt(l=l, average=average, wl=wl, weighted=weighted)
        featl.compute(system=(box, positions), neighbors=nlist)
        features[descriptor+'{}'.format(l)] = getattr(featl,what)
        order[descriptor+'{}'.format(l)] = featl.order
    return order, features

def get_feat_metal(box, positions, at_num, structure, values, average, weighted, descriptor, what):
    aq = freud.locality.AABBQuery(box,positions)
    nlist=aq.query(positions,{'mode':'nearest','num_neighbors':4}).toNeighborList()
    features = {}
    order = {}
    wl=False
    if descriptor=='w':
        wl=True
    for l in values:
        featl = freud.order.Steinhardt(l=l, average=average, wl=wl, weighted=weighted)
        featl.compute(system=(box, positions), neighbors=nlist)
        features[descriptor+'{}'.format(l)] = getattr(featl,what)
        order[descriptor+'{}'.format(l)] = featl.order
    return order, features

# Compute Bond Order parameters
def get_bondorder(box, positions, structure, bins):
    bondorder = freud.environment.BondOrder(bins)
    bondorder_arr = bondorder.compute(system=(box, positions), neighbors={'num_neighbors': 10}).bond_order
    
    return bondorder_arr

#___INPUT DEFINITION___
#
# Define here what you want to calculate
bins=3
bondorder=False
steinhardt=True
# Specify whether Steinhardt parameters are going to be evaluated for metals
# only too 
tetra=False
values=[2,4,6,8,10,12,14]
average=False
weighted=False
descriptor='q'
# Create a prefix to indicate whether parameters are averaged or not
prefix=''
if average:
    prefix='average_'
if weighted:
    prefix +='weighted_'
# Dict that stores 'Ql' Steinhard parameters, as 'name':'ql', being l the number
structure_feat={}
# Dict containing the system wide normalization of the Ql/Wl order parameter
structure_order={}
# Dict containing the bond order parameter
structure_bondorder={}
# Plotting reasons: find inferior and superior limits for the descriptor values, and use them in the plot
range_min=[]
range_max=[]
if steinhardt:
# For each 'name', i.e.: VASP optimized structure, call the function get_features and calculate Voronoi
# neighbors and Ql
    for name, (box, positions,index,at_num,symbols) in structures.items():
        #structure_order[name], structure_feat[name] = get_feat(box, positions, name, values, average, weighted, descriptor, 'particle_order')
        structure_order[name], structure_feat[name] = get_feat_metal(box, positions, at_num, name, values, average, weighted, descriptor, 'particle_order')
elif bondorder:
    for name, (box, positions,index,at_num,symbols) in structures.items():
        structure_bondorder[name] = get_bondorder(box, positions, name, bins)
    # Print name, number of neighbors and neighbors
    #print(name,len(structure_nlist[name].neighbor_counts),structure_nlist[name].neighbor_counts)
    #print(name)
    #print(structure_feat[name])
    #print()

if bondorder:
    for name in structure_order.keys():
        print(name,structure_bondorder[name][0])
     
if steinhardt:
# Write the system wide normalization of the ql/wl order parameters to a file
    filename=prefix+descriptor
    orderfile=filename+'_order.txt'
    with open(orderfile, 'w') as f:
    # Write header with the names of the columns
        f.write('Structure\t')
        for l in values:
            f.write(descriptor+str(l)+' ')
        f.write('\n')
    # Write the name of the structure and the calculated values
        for name in structure_order.keys():
            f.write(name+'\t')
            lists = [ structure_order[name][descriptor+str(l)] for l in values ]
            for i in range(len(values)):
                f.write(str(lists[i])+' ')
            f.write('\n')
    f.close()
#print(range_min=np.amin(range_min))
#print(range_max=np.amax(range_max))
# Trial to find neighbors from a cell (box and positions)
#query_args=dict(mode='nearest',num_neighbors=4,exclude_ii=True)
#neighs=freud.locality.AABBQuery(box,positions).query(positions,query_args).toNeighborList()
#print(len(structure_features[name].distances),structure_features[name].distances)

    for l in values:
        fname=filename+'{}'.format(l)
        plt.figure(figsize=(5, 5), dpi=200)
        for name in structures.keys():
            plt.hist(structure_feat[name][descriptor+str(l)], bins=20, label=name, alpha=0.7)
#plt.hist(structure_feat[name][calc], range=(range_min, range_max), bins=80, label=name, alpha=0.7)
#plt.title(r'$q_{{{l}}}$'.format(l=l))
        plt.ylabel("Frequency", fontsize=14)
        plt.xlabel('$q_{%i}$' % (l), fontsize=14)
        plt.legend(fontsize='xx-small')
        plt.title(fname)
        plt.savefig(fname)
#for lh in plt.legend().legendHandles:
#    lh.set_alpha(1)
#plt.show()
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

