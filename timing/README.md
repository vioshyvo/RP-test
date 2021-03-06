## Query time and accuracy comparisons

Comparison of query and index building time between the new and the old
version of mrpt.

Start by downloading the data sets:
```
source ../../py27/bin/activate
./get_data.sh
```

Edit the directory containing the header file `Mrpt.h` you want to test to the `mrpt_tester/Makefile`, for example:
```
MRPT_PATH=../../../mrpt/cpp
```

Edit the parameters you want to use for test to `parameters/mnist.sh` (the flag `PARALLEL` controls whether OpenMP threading is allowed), and run the comparison script:
```
./comparison.sh mnist
```
or
```
./comparison.sh mnist <postfix>
```
The results are printed into the files `results/mrpt_<postfix>.txt` (or plain `results/mrpt.txt` if you did not assign any postfix) and `results_mrpt_old.txt`. The format of the result files is
```
<k> <n_trees> <depth> <density> <vote threshold> <mean recall> <sd. of recall>  <query time (total)> <sd. of query times> <index building time>.  
```
where query time is the combined time for all the test points, for example 100 points. Values of `k` used are `k=1,10,100`. These are hard-coded into the `mrpt_tester/tester.cpp` and `mrpt_old_tester/tester.cpp` and `comparison.sh` (for exact search).

Finally, plot the results, for example for `k=10`:
```
python2 plot.py 10 results/mnist/mrpt.txt results/mnist/mrpt_old.txt
```
