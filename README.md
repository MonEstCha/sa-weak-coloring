# sa-weak-coloring
Simulated Annealing for Weak Coloring

This repository contains the source code for the Simulated Annealing (SA) procedure and the Python script for analyzing several logfiles or a single one. Logfiles are written by the SA procedure at the end of a run. 
Note: A directory 'log' should be present in the working directory or the path to the logfile be adapted.
The files Header, ReadText, FilesOp and FlagParser were taken from Nadara et al. (2019) and adapted for the purposes of this Bachelor thesis where necessary.  
The file suffix v1 denotes a version of SA in which random pairs of vertices are swapped in the current vertex order; in v2 vertices with a maximum weakly reachable set size are moved to a lower position in the order.
