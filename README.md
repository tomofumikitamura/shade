# SHADE
Python implementation of SHADE.

# Evaluation
Experimental settings are as follows.
+ Population 100, archive rate 2, pbest rate 0.1 and history sizes H = 15.

Comparison of python-SHADE and Tanabe-SHADE (CEC2014 benchmarks, D = 30 dimension).
Mean of best objective function values found (48 runs).
The +/-/≈ symbols indicate result of Wilcoxon ranked-sum test vs. baseline SHADE (p=0.05) (+: better, -: worse, ≈: no significant difference).
|     |   |   TEST    |   SHADE   |
|-----|---|-----------|-----------|
| F1  | = | 2.479e+04 | 1.390e+02 |
| F2  | = | 0.000e+00 | 0.000e+00 |
| F3  | = | 0.000e+00 | 0.000e+00 |
| F4  | - | 5.881e+01 | 1.321e+00 |
| F5  | = | 2.011e+01 | 2.010e+01 |
| F6  | = | 1.764e+00 | 9.190e-01 |
| F7  | = | 8.210e-04 | 3.594e-04 |
| F8  | = | 0.000e+00 | 0.000e+00 |
| F9  | = | 1.437e+01 | 1.366e+01 |
| F10 | + | 0.000e+00 | 5.205e-03 |
| F11 | = | 1.404e+03 | 1.432e+03 |
| F12 | - | 1.657e-01 | 1.567e-01 |
| F13 | = | 2.107e-01 | 2.126e-01 |
| F14 | = | 2.562e-01 | 2.461e-01 |
| F15 | = | 2.500e+00 | 2.498e+00 |
| F16 | = | 9.192e+00 | 9.162e+00 |
| F17 | = | 9.969e+02 | 1.030e+03 |
| F18 | - | 6.371e+01 | 5.066e+01 |
| F19 | - | 8.717e+00 | 4.602e+00 |
| F20 | = | 1.551e+01 | 1.439e+01 |
| F21 | = | 2.823e+02 | 2.735e+02 |
| F22 | = | 9.228e+01 | 9.961e+01 |
| F23 | + | 3.149e+02 | 3.152e+02 |
| F24 | = | 2.275e+02 | 2.266e+02 |
| F25 | + | 2.028e+02 | 2.032e+02 |
| F26 | = | 1.002e+02 | 1.002e+02 |
| F27 | = | 3.314e+02 | 3.286e+02 |
| F28 | + | 4.302e+02 | 8.477e+02 |
| F29 | + | 2.119e+02 | 7.236e+02 |
| F30 | + | 5.645e+02 | 1.979e+03 |

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
