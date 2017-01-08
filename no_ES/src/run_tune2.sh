rm ./err/*
rm ./out/*

bsub -W 2400 -n 10 -o ./out/out.%J -e ./err/err.%J mpiexec -n 10 /share2/zyu9/miniconda/bin/python2.7 tune_LDA.py exp

