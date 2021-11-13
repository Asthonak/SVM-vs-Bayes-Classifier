#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>

using namespace Eigen;
using namespace std;


float squared(Vector2f x)
{
    return x.transpose() * x;
}


float discriminant_case_1(VectorXf x, VectorXf mu, float sigma, float prior_prob)
{
	float discriminant = (((1.0/sigma) * mu).transpose() * x) + (-1.0 / (2*sigma)) * squared(mu);
	if(prior_prob != 0.5)
	{
		discriminant += log(prior_prob);
	}
	return discriminant;
}

/*Implementation of case 3 of Bayesian Classifier*/
float discriminant_case_3(VectorXf x, VectorXf mu, MatrixXf sigma, float prior_prob)
{
	float discriminant = (x.transpose() * (-0.5 * sigma.inverse()) * x) 
			   + ((sigma.inverse() * mu).transpose() * x)(0) + (-0.5 * mu.transpose() * sigma.inverse() * mu) 
			   + (-0.5 * log(sigma.determinant()));
	if(prior_prob != 0.5)
	{
		discriminant += log(prior_prob);
	}
	return discriminant;
}

/*makes classification based on case 3*/
int classify_case_3(VectorXf x, VectorXf mu1, MatrixXf sigma1, VectorXf mu2, MatrixXf sigma2, float prior_prob1 = 0.5, float prior_prob2 = 0.5)
{
	float discriminant1 = discriminant_case_3(x, mu1, sigma1, prior_prob1);
	float discriminant2 = discriminant_case_3(x, mu2, sigma2, prior_prob2);

	if (discriminant1 > discriminant2)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}


float discriminant_min_distance(VectorXf x, VectorXf mu)
{
	return -1.0 * squared(x-mu);
}

/*Gets ML mean*/
VectorXf ml_mean(vector<VectorXf> x)
{
	VectorXf mean;
	
	if(x.size() <= 0)
	{
		return mean;
	}
	mean = VectorXf::Zero(x[0].rows());
	for(vector<int>::size_type i = 0; i < x.size(); i++)
	{
		mean += x[i];
	}
	return mean / x.size();
}

/*Gets ML covariance*/
MatrixXf ml_covariance(vector<VectorXf> x, VectorXf mean)
{
	
	MatrixXf covariance;

	if (x.size() <= 0)
	{
		return covariance;
	}
	covariance = MatrixXf::Zero(x[0].rows(), x[0].rows());

	for (vector<int>::size_type i = 0; i < x.size(); i++)
	{
		covariance += (mean - x[i])*((mean - x[i]).transpose());
	}
	return covariance / x.size();
}

bool threshold_case_3(VectorXf x, VectorXf mu, MatrixXf sigma, float threshold)
{
	return (-0.5 * (x - mu).transpose() * sigma.inverse() * (x - mu)) > threshold;
}
