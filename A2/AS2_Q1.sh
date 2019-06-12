#!/bin/bash

#$ -l h_rt=4:00:00  #time needed

#$ -pe smp 20 #number of cores

#$ -l rmem=15G #number of memery

#$ -P rse-com6012

#$ -q rse-com6012.q

#$ -j y

#$ -M mwziwu1@sheffield.ac.uk

#$ -m bea

#$ -o COM6012_AS2Q1.output

module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
source activate myspark
spark-submit --driver-memory 10g --executor-memory 5g --master local[20] AS2_Q1.py
