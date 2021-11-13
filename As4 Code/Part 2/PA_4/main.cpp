#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include "bayes.cpp"
#include <cstdlib>
#include <string>
#include <fstream>
#include <sstream>

using namespace Eigen;
using namespace std;

/*
USEFULL FUNCTIONS:

VectorXf ml_mean(vector<VectorXf> x)
MatrixXf ml_covariance(vector<VectorXf> x, VectorXf mean)
int classify_case_3(VectorXf x, VectorXf mu1, MatrixXf sigma1, VectorXf mu2, MatrixXf sigma2, float prior_prob1, float prior_prob2) ***returns 1 for class 1 and returns 2 for class 2***
*/



int main(){
	
	classify_case_3();
	
	return 0;

}
