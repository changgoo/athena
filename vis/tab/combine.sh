#!/bin/bash

# Define the base name.
BASE=$1
OUTID=$2
PARID=$3

# Find the number of blocks.
read nblocks < <(ls $BASE.block*.$OUTID.00000.$PARID.tab | wc -l)

# Find the number of snapshots.
read ntimes < <(ls $BASE.block0.$OUTID.*.$PARID.tab | wc -l)
#(( --ntimes ))

for i in $(seq -f "%05g" 0 $((ntimes - 1))); do
	if (( nblocks > 1 )); then
		cat $BASE.block*.$OUTID.$i.$PARID.tab | \
                sort --general-numeric-sort | \
                sed "2,$((nblocks*2-1))d" > $BASE.$OUTID.$i.$PARID.tab
	fi
done

#find . -name '$BASE.block*.$OUTID.*.$PARID.tab'
