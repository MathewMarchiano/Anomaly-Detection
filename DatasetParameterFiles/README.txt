The files are used for the AnomalyDetection module. Read the"ParameterValueFile_TEMPLATE.txt" file
to see what components are required in order for the Anomaly Detection program to work.

Files with "LesserRandomPack" in their name are just like every other parameter file, except
instead of having 3 codebooks of varying codeword length, there are 3 codebooks containing
codewords of all the same length; however, the row and column Hamming distances vary. The
codebooks are arranged in order of lowest minimum Hamming distance to highest minimum distance.
(i.e. codebook 1 has the lowest min. HD, and codebook n has the highest).

NOTE: If using any of the already created parameter value files, you have to update the file paths
to those that reflect where you want the data to be stored. You also have to update the CSV file path.