#! /bin/tcsh
# chmod +x run_hpc.sh
rm err/*
rm out/*
foreach VAR (`seq 0 1 30`)
  bsub -q standard -W 1500 -n 2 -o ./out/$VAR.out.%J -e ./err/$VAR.err.%J /share/tjmenzie/zyu9/miniconda2/bin/python2.7 runner.py error_hpcc $VAR
end

