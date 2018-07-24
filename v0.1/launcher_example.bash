#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

# example: focusing of SHARAD radargram X
#	- all inputs must be generated first from EDR files (and SPICE kernels for refined orbit parameters)
#	- all mentioned subfolders must be created before launching
#	- base filename for inputs:  radargram_X_input
#	- base filename for outputs: radargram_X_output

# generate radargram X subset from frame 10000 to frame 20000 using US-SHARAD Science Team style parameters
python3 sofa.py -ip radargram_X_folder/rdr/ -if radargram_X_input -op radargram_X_folder/output/ -of radargram_X_output -im mola_radius_folder/r128_44-88n -d -p 5 -fs 10000 -fe 20000 -fk 26 -gn 0

# generate the same radargram subset, but using short squinted apertures from -1.8 to 1.8 degrees squint
python3 sofa.py -ip radargram_X_folder/rdr/ -if radargram_X_input -op radargram_X_folder/output/ -of radargram_X_output -im mola_radius_folder/r128_44-88n -d -p 4 -fs 10000 -fe 20000 -fk 26 -a 128 -gn 0
for i in {5..180..5}
  do 
     k=`echo $i / 100 | bc -l`
     python3 sofa.py -ip radargram_X_folder/rdr/ -if radargram_X_input -op radargram_X_folder/output/ -of radargram_X_output -im mola_radius_folder/r128_44-88n -d -p 4 -fs 10000 -fe 20000 -fk 26 -a 128 -gn 0 -s -$k
     python3 sofa.py -ip radargram_X_folder/rdr/ -if radargram_X_input -op radargram_X_folder/output/ -of radargram_X_output -im mola_radius_folder/r128_44-88n -d -p 4 -fs 10000 -fe 20000 -fk 26 -a 128 -gn 0 -s $k
done


