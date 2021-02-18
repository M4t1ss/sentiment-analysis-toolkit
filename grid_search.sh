#!/bin/bash


for lrate in 0.00005 0.00001 0.000005
do
	for drop in 0.2 0.3 0.4 0.5
	do
		echo "Learning rate - ${lrate}"
		echo "Dropout - ${drop}"
		python train.py --lr=$lrate --dropout=$drop
	done
done
