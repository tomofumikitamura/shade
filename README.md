# SHADE
Python implementation of SHADE.

# Requirements
+ Python3.X
+ Pybind11 https://github.com/pybind/pybind11
+ CEC2014 Benchmark suit part A https://github.com/P-N-Suganthan/CEC2014

# Usage
Edit cec14_test_func.cc as follows.
```c
//Add pybind11 header
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

//void cec14_test_func(double *x, double *f, int nx, int mx,int func_num)
void cec14_test_func(double *x,  std::vector<double> &f, int nx, int mx, int func_num)

//Add python bind function
std::vector<double> cec14_test(int func, int dim, std::vector<double> &v){
    int mx = v.size()/dim;
    int size = v.size();
    std::vector<double> fx(mx);
    double x[size];
    for(int i=0;i<size;++i) x[i]=v[i];
    cec14_test_func(x, fx, dim, mx, func);
    return fx;
}

//Define Python library
PYBIND11_PLUGIN(cec14) {
    pybind11::module m("cec14", "mylibs made by pybind11");
    m.def("cec14_test", &cec14_test);
    return m.ptr();
}
```
Compile test function. See https://github.com/pybind/pybind11.

# Reference
+ Ryoji Tanabe and Alex Fukunaga: Success-History Based Parameter Adaptation for Differential Evolution, Proc. IEEE Congress on Evolutionary Computation (CEC2013)
+ Ryoji Tanabe and Alex Fukunaga: Evaluating the performance of SHADE on CEC 2013 benchmark problems, Proc. IEEE Congress on Evolutionary Computation (CEC2013)

Tanabe's original (C++/MATLAB) implementation is in https://ryojitanabe.github.io/publication
