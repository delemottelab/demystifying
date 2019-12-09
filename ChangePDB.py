import argparse
import sys
import numpy as np
import re
import matplotlib.pyplot as plotter
import os

parser = argparse.ArgumentParser(epilog='Change PDB files, eg the beta field. Annie Westerlund 2017.');
parser.add_argument('-top','--topology_file',help='Input 1 topology file (.gro, .pdb, etc)',type=str,default='');
parser.add_argument('-beta','--change_beta',help='Flag for writing beta of residues (optional).',action='store_true');
parser.add_argument('-hel_beta','--change_helix_beta',help='Flag for writing beta of helices (optional).',action='store_true');
parser.add_argument('-rename','--rename_atoms',help='Flag for renaming residues from CHARMM-gui to Gromacs (optional).',action='store_true');

parser.add_argument('-renumber','--renumber_residues',help='Flag for renumbering residues from sequential to chain numbering. Need to enter number of chains or set first resID. (optional)',action='store_true');
parser.add_argument('-n_chains','--number_of_chains',help='Number of chains to use when renumbering the residues',default=1);
parser.add_argument('-first_resid','--first_resID',help='Start resID in the structure.',default=1);

parser.add_argument('-f','--in_file',help='Input file.',default='');
parser.add_argument('-f_conv','--helix_ind_conv_file',help='File containing residue number of helices (optional).',default=0);
parser.add_argument('-fe','--file_end_name',type=str,help='Output file end name (optional)', default='');
parser.add_argument('-od','--out_directory',type=str,help='The directory where data should be saved (optional)',default='');
parser.add_argument('-o','--out_file',help='File name (optional).',default='structure');

args = parser.parse_args();

# Put / at end of out directory if not present. Check so that folder extsts, otherwise construct it.
if args.out_directory !='':
	if args.out_directory[-1] != '/':
		args.out_directory += '/';

	if not os.path.exists(args.out_directory):
		os.makedirs(args.out_directory);

# Change residue names
if args.rename_atoms:
	print('Changing residue names to match gromacs names');
	fID1 = open(args.topology_file,'r');
	fID2 = open(args.out_directory + args.out_file + '_gmx_' + args.file_end_name+'.pdb','w');
	first_line = True;
	
	for line in fID1:
		if line[0:3] == 'END':
			continue;
		
		if first_line:
			first_line = False;
			fID2.write(line);
		else:
			atom = line[13:17];
			residue = line[17:21];
			
			#Change the names of atoms and residues
			if atom == 'CL  ':
				atom = 'CLA ';
				residue = 'CLA ';
			elif atom == 'K   ':
				atom = 'POT ';
				residue = 'POT ';
			elif atom == 'OH2 ':
				atom = 'OW  ';
				residue = 'SOL ';
			elif atom == 'H1  ':
				atom = 'HW1 ';
				residue = 'SOL ';
			elif atom == 'H2  ':
				atom = 'HW2 ';
				residue = 'SOL ';
			
			tmpLine = line[0:13] + atom + residue + line[21::];
			fID2.write(tmpLine);
	
	fID1.close();
	fID2.close();


# Change beta columns general
if args.change_beta:
	print('Changing beta column ' + args.file_end_name);
	values = np.loadtxt(args.in_file);
	print('Max value: '+str(np.max(values)));
	print('Min value: '+str(np.min(values)));
	

	counter = 0;
	resid_counter = -1;
	current_resid = -1;
	next_resid_write = -1;
	fID1 = open(args.topology_file,'r');
	fID2 = open(args.out_directory + args.out_file + '_beta_'+ args.file_end_name+'.pdb','w');
	first_line = True;
	
	for line in fID1:
		if line[0:3] == 'END':
			continue;
		
		if first_line:
			first_line = False;
			fID2.write(line);
		else:
			if line[0:3] == 'END' or line[0:3]=='TER':
				continue;
			resid = int(line[23:26]);
			
			if resid != current_resid:
				current_resid = resid;
				resid_counter += 1;
			
			if resid_counter < len(values):
				fID2.write(line[0:60]);
				fID2.write(' ');
				fID2.write('%.3f' % values[resid_counter]);
				fID2.write(line[66::]);
			else:
				fID2.write(line[0:60]);
				fID2.write(' ');
				fID2.write('0.00');
				fID2.write(line[66::]);
				
	
	fID1.close();
	fID2.close();

# Change beta column of helices
if args.change_helix_beta:
	print('Changing beta column for each helix');
	helix_resid_convert_ind = np.loadtxt(args.helix_ind_conv_file);
	
	values = np.loadtxt(args.in_file);
	values -= np.min(values);
	values /= np.max(values);
	values = np.floor(1000*values)/100;
	
	counter = 0;
	resid_counter = -1;
	current_resid = -1;
	current_resid_min = helix_resid_convert_ind[counter,0];
	current_resid_max = helix_resid_convert_ind[counter,1];
	fID1 = open(args.topology_file,'r');
	fID2 = open(args.out_directory + args.out_file + '_helix_beta_'+ args.file_end_name+'.pdb','w');
	first_line = True;	
	
	for line in fID1:
		if line[0:3] == 'END' or line[0:3]=='TER':
			continue;
		
		if first_line:
			first_line = False;
			fID2.write(line);
		else:
			resid = int(line[23:26]);
			if resid != current_resid:
				current_resid = resid;
				resid_counter += 1;

			if resid_counter >= current_resid_min and resid_counter <= current_resid_max:
				fID2.write(line[0:62]);
				fID2.write(str(values[counter]));
				fID2.write(line[66::]);
			else:
				fID2.write(line[0:62]);
				fID2.write('0.00');
				fID2.write(line[66::]);
				if resid_counter > current_resid_max:
					counter += 1;
					if counter < len(helix_resid_convert_ind[::,0]):
						current_resid_min = helix_resid_convert_ind[counter,0];
						current_resid_max = helix_resid_convert_ind[counter,1];
					else: 
						current_resid_min = -1;
						current_resid_max = -1;

	
	fID1.close();
	fID2.close();

if args.renumber_residues:

	chain_letters = 'ABCDEFGHIJKLMNOPQRSTUVXYZ'
	
	n_chains = float(args.number_of_chains);
	first_resid = int(args.first_resID);
	
	fID1 = open(args.topology_file,'r');
	fID2 = open(args.out_directory + args.out_file + '_renumbered_'+ args.file_end_name+'.pdb','w');
	
	lines = [];	
	# Collect all lines
	for line in fID1:
		lines.append(line);
	
	# Count the number of residues
	counter = 0;
	resid_counter = 0;
	current_resid = -1;
	first_line = True;
	
	
	for i in range(1,len(lines)):
		line = lines[i];		

		if line[0:3] == 'END' or line[0:3]=='TER':
			continue;


		resid = int(line[23:26]);
		if resid != current_resid:
			current_resid = resid;
			resid_counter += 1;
	
	# Compute total number of residues per chain
	nResiduesPerChain = np.floor(resid_counter/n_chains);
	
	# Renumber residues
	counter = 0;
	resid_counter = 0;
	current_resid = -1;
	first_line = True;
	new_resid = first_resid-1;
	chain_id = 0;
	
	for i in range(0,len(lines)):
		line = lines[i];
		if line[0:3] == 'END' or line[0:3]=='TER':
			fID2.write(line);
			continue;
		
		if first_line:
			first_line = False;
			fID2.write(line);
			continue;
		
		resid = int(line[22:26]);
		if resid != current_resid:
			current_resid = resid;
			resid_counter += 1;
			new_resid += 1;
			if resid_counter > nResiduesPerChain:
				new_resid = first_resid;
				resid_counter = 1;
				chain_id += 1;
				fID2.write('TER\n');
		
		if new_resid < 10:
			fID2.write(line[0:21]+chain_letters[chain_id]+'   '+str(new_resid)+line[26::]);
		elif new_resid < 100:
			fID2.write(line[0:21]+chain_letters[chain_id]+'  '+str(new_resid)+line[26::]);
		else:
			fID2.write(line[0:21]+chain_letters[chain_id]+' '+str(new_resid)+line[26::]);

	print(nResiduesPerChain)


