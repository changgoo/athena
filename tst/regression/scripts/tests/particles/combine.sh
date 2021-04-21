#!/bin/bash

# Define the base name.
BASE=UniStream
OUTPUT=$BASE.par.dat

# Find the number of blocks.
read nblocks < <(ls $BASE.block*.out2.00000.par0.tab | wc -l)

# Find the number of snapshots.
read ntimes < <(ls $BASE.block0.out2.*.par0.tab | wc -l)
(( --ntimes ))

for i in $(seq -f "%05g" 0 $((ntimes - 1))); do
	if (( nblocks > 1 )); then
		cat $BASE.block*.out2.$i.par0.tab | \
                sort --general-numeric-sort | \
		sed "2,${nblocks}d" > $BASE.$i.par.tab
	else
		mv $BASE.block0.out2.$i.par0.tab $BASE.$i.par.tab
	fi
	cat $BASE.block*.out1.$i.tab > $BASE.$i.tab
done

find . -name $BASE'.block*.out1.*.tab' -delete
find . -name $BASE'.block*.out2.*.tab' -delete
