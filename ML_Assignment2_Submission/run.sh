
#!/bin/bash
set -e
set -x


#python Assignment2.py -fold 1 data/a5a.train -sanity data/table2 -fold 1 data/a5a.train -test data/a5a.test -q32 -q33 -q3grad
python Assignment2.py -fold 1 data/a5a.train -sanity data/table2 -fold 1 data/a5a.train -test data/a5a.test |& tee q31
python Assignment2.py -fold 1 data/a5a.train -sanity data/table2 -fold 1 data/a5a.train -test data/a5a.test -q32 |& tee q32
python Assignment2.py -fold 1 data/a5a.train -sanity data/table2 -fold 1 data/a5a.train -test data/a5a.test -q33 |& tee q33
python Assignment2.py -fold 1 data/a5a.train -sanity data/table2 -fold 1 data/a5a.train -test data/a5a.test -q3grad |& tee q3grad
