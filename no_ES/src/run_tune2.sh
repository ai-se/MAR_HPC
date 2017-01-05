rm ./err/*
rm ./out/*

bsub -W 12000 -n 10 -o ./out/out.%J -e ./err/err.%J /share2/zyu9/miniconda/bin/python2.7 tune_LDA.py exp

