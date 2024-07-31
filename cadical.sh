#!/bin/sh
#PBS -v arg

#PBS -N cadical 

## we can use short as long as the timeout is less than 3 hours. 
##PBS -q  short_l_p
#PBS -q  zeus_new_q

# Send the e-mail messages from the server to a user address
# This line and the Technion address are mandatory!
#--------------------------------------------------------
#PBS -M fre.gilad@campus.technion.ac.il

##PBS -mbea  // default is -ma, which is only crashes	
#
# running 1 process on the available CPU of the available node 
#------------------------------------------------------------------
#PBS -l select=1:ncpus=1

# running job on a node exclusively
##PBS -l place=excl 

# requesting wall time
#PBS -l walltime=1:00:00
##PBS -l cput=0:10:00

PBS_O_WORKDIR=$HOME/source/cadical-simd
cd $PBS_O_WORKDIR

#Run command
#-----------------------
## if we add pipe to grep "###" it doesn't work on timeouts. 
./build/cadical $arg > $out


