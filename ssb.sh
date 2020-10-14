#!/bin/bash
#
for i in *$1*.sh
do
 echo 'sbatch '$i
 #rm $i
 chmod +x $i
 sbatch $i
 #sleep 10s
done
