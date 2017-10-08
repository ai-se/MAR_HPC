#! /bin/tcsh

foreach VAR (`seq 1 1 30`)
  bsub -q standard -W 1000 -n 2 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share2/zyu9/miniconda/bin/python2.7 runner.py rerror_hpcc $VAR
end

