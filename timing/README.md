## Query time and accuracy comparisons

Comparison of query and index building time between the new and the old
version of mrpt.

Start by downloading the mnist data set:
```
./get_mnist.sh
```

Edit the directory containing the header file `Mrpt.h` you want to test to the `mrpt_tester/Makefile`, for example:
```
MRPT_PATH=../../../mrpt/cpp
```

Edit the parameters you want to use for test to `parameters/mnist.sh`, and run the comparison script:
```
./comparison.sh mnist
```
The results are printed into the files `results/mrpt.txt` and `results_mrpt_old.txt`. The format of the result files is
```
k n_trees depth density v recall recall.sd query.time build.time,  
```
where query time is the combined time for all the test points, for example 100 points. Values of `k` used are `k=1,10,100`. These are hard-coded into the `mrpt_tester/tester.cpp` and `mrpt_old_tester/tester.cpp` and `comparison.sh` (for exact search). 

Finally, plot the results, for example for `k=10`:
```
python2 plot.py 10 results/mnist/mrpt.txt results/mnist/mrpt_old.txt
```
