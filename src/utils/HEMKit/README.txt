This directory contains the software of HEMKit, a collection of hierarchical 
evaluation measures. The software was developed for the  paper: 

Aris Kosmopoulos, Ioannnis Partalas, Eric Gaussier, George Paliouras 
and Ion Androutsopoulos, "Evaluation Measures for Hierarchical Classification: 
a unified view and novel approaches"

This kit is implemented in C++. Its source code can be compiled 
with GCC, using the command make in the software directory. 

The software of the filter is released with the GNU General Public 
License; please consult the file COPYING.txt for more information. 

*********************************************************************
**** THIS SOFTWARE IS A RESEARCH PROTOTYPE AND IS PROVIDED WITH *****
****    ABSOLUTELY NO GUARANTEE AND ABSOLUTELY NO SUPPORT!      *****
*********************************************************************

Usage
=====

After compilation an executable with the name HEMKit will be created
in the bin folder. This executable can be run as follows:

HEMKit -hierarchy -truecat -predcat -maxdist -maxerr [-auxfile]

The arguments of the executable are as follows:

-hierarchy 
A text file containing the hierarchy. The hierarchy file contains 
the hierarchy information about the categories of the train set. Each line 
of this file is a relation between a parent and a child node. For example, 
the line:

10 20

is to be read as node 10 is parent of node 20. 

-truecat 
A text file containing in each line the true categories of an 
instance separated by spaces. For example:

1 2 3
1 4 5
2 4

denotes that the first instance belongs to categories 1, 2 and 3,
the second instance belongs to categories 1, 4 and 5,
and the third instance belongs to categories 2 and 4.

-predcat
A text file containing in each line the predicted categories of an 
instance separated by spaces. For example:

1
2 4
2 4

denotes that the first instance is predicted to belong to category 1,
the second instance is predicted to belong to categories 2 and 4,
and the third instance is predicted to belong to categories 2 and 4.

-maxdist
The maximum distance that the measures will search in order 
to link nodes. Above that threshold all nodes will be considered to have a 
common ancestor. In the extreme case that this threshold is set to 1 all nodes 
are considered  to have a dummy common ancestor as direct ancestor of their ancestors. 
This threshold should usually be set to as large a number as possible, so that
the true distances are calculated. But in very large datasets it may 
be set to low values, like 4 or 5, for computational reasons (see paper for 
further details).

-maxerr
Specifies the maximum error with which pair-based measures penalize
nodes that are matched with the default class (see paper for further details).
In our experiment this is usually set to 5.

-auxfile 
The name of an auxiliary file storing data for statistical significance tests. 
Each line of this file contains the results of all evaluation measures for an 
instance (not averaged over all instances). 

Example:
After running make in the software folder, go to the example files folder and execute 
the following command:

./bin/HEMKit cat_hier.txt Golden.txt result.txt 100000 5

This will use the cat_hier.txt file as a hierarchy,
the Golden.txt as the file containing the true categories per instance,
the result.txt as the file containing the predicted categories per instance,
will not use any dummy common ancestors since 100000 is a very large number 
compared to the hierarchy, and will penalize nodes that were matched with a 
default one with 5. Since -auxfile is not used, no extra file will be created.
The results of each measure will be printed to the standard output.

Source code
===========

The source code can be found in the following directory:

software sub-directory

Copyright (c) 2013 A. Kosmopoulos.
Please send bug reports to Aris Kosmopoulos <akosmo@iit.demokritos.gr>.

------ END OF FILE ------
