#! /bin/tcsh

foreach VAR (`seq 1 1 30`)
  bsub -q standard -W 1000 -n 2 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J python runner.py rerror_hpcc $VAR
end

