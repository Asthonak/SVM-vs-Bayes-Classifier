#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <sstream>

using namespace Eigen;
using namespace std;


float squared(Vector2f x)
{
    return x.transpose() * x;
}

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

float discriminant_case_1(VectorXf x, VectorXf mu, float sigma, float prior_prob)
{
	float discriminant = (((1.0/sigma) * mu).transpose() * x) + (-1.0 / (2*sigma)) * squared(mu);
	if(prior_prob != 0.5)
	{
		discriminant += log(prior_prob);
	}
	return discriminant;
}

//Vector2F
float discriminant_case_3(Vector2f x, Vector2f mu, Matrix2f sigma, float prior_prob)
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

//VectorXf
float discriminant_case_3(VectorXf x, VectorXf mu, MatrixXf sigma, float prior_prob)
{

	float discriminant = (x.transpose() * (-0.5 * sigma.inverse()) * x) 
			   + ((sigma.inverse() * mu).transpose() * x)(0) + (-0.5 * mu.transpose() * sigma.inverse() * mu);
			   //+ (-0.5 * log(sigma.determinant()));
	if(prior_prob != 0.5)


	{
		discriminant += log(prior_prob);
	}
	
	return discriminant;
}


void ReadEigenFaceData(vector<VectorXf>& male, vector<VectorXf>& female, string fname, string targetname)
	{
		ifstream faces(fname);
		ifstream target(targetname);

		string ln;
		getline(target, ln);

		istringstream tstream(ln);

		while (getline(faces, ln))
		{
			istringstream iss(ln);

			vector<float> vals;

			float val;
			while (iss >> val)
			{
				vals.push_back(val);
			}

			VectorXf f(vals.size());

			for (int i = 0; i < vals.size(); i++)
			{
				f.row(i) << vals[i];
			}

			int classification = 0;
			tstream >> classification;

			if (classification == 1)
				male.push_back(f);
			else
				female.push_back(f);
		}
	}


//Vector2f
int classify_case_3(Vector2f x, Vector2f mu1, Matrix2f sigma1, Vector2f mu2, Matrix2f sigma2, float prior_prob1 = 0.5, float prior_prob2 = 0.5)
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

//VectorXf
void classify_case_3()
{
/*
Swap 16_20 and 48_60 for different data sets
*/
	for (int i = 1; i < 4; i++)
	{
		cout << "Fold " << i << ":" << endl;

		//Training faces
		vector<VectorXf> maleTrainingFaces;
		vector<VectorXf> femaleTrainingFaces;

		char infname[256];
		sprintf(infname, "48_60_reformatted_data/trPCA_0%i-new.txt", i);

		char tgfname[256];
		sprintf(tgfname, "48_60_reformatted_data/TtrPCA_0%i.txt", i);

		ReadEigenFaceData(maleTrainingFaces, femaleTrainingFaces, infname, tgfname);

		//cout << "mf size: " << maleTrainingFaces.size() << endl;
		//cout << "ff size: " << femaleTrainingFaces.size() << endl;

		VectorXf mm = ml_mean(maleTrainingFaces);
		MatrixXf mc = ml_covariance(maleTrainingFaces, mm);

		VectorXf fm = ml_mean(femaleTrainingFaces);
		MatrixXf fc = ml_covariance(femaleTrainingFaces, fm);

		//Test Faces
		vector<VectorXf> maleTestFaces;
		vector<VectorXf> femaleTestFaces;

		sprintf(infname, "48_60_reformatted_data/tsPCA_0%i-new.txt", i);

		sprintf(tgfname, "48_60_reformatted_data/TtsPCA_0%i.txt", i);

		ReadEigenFaceData(maleTestFaces, femaleTestFaces, infname, tgfname);

		int count = 0;
		int maleCorrect = 0;
		int femaleCorrect = 0;

		//Here we do P(w1)=P(w2)=0.5 -----------------------------------------------------------
		//cout << "classifying male faces" << endl;

		for (int i = 0; i < maleTestFaces.size(); i++)
		{
			count++;
			float d1 = discriminant_case_3(maleTestFaces[i], mm, mc, 0.5);
			float d2 = discriminant_case_3(maleTestFaces[i], fm, fc, 0.5);

			//cout << "face: " << maleTrainingFaces[i] << endl;
			//cout << "d1: " << d1 << endl;
			//cout << "d2: " << d2 << endl;

			if (d1 >= d2)
			{
				maleCorrect++;
			}
		}

		//cout << "classifying female faces" << endl;

		for (int i = 0; i < femaleTestFaces.size(); i++)
		{
			count++;
			float d1 = discriminant_case_3(femaleTestFaces[i], fm, fc, 0.5);
			float d2 = discriminant_case_3(femaleTestFaces[i], mm, mc, 0.5);

			if (d1 >= d2)
			{
				femaleCorrect++;
			}
		}

		//Validatoin Faces (essentially more test faces)
		vector<VectorXf> maleValidationFaces;
		vector<VectorXf> femaleValidationFaces;

		sprintf(infname, "48_60_reformatted_data/valPCA_0%i-new.txt", i);

		sprintf(tgfname, "48_60_reformatted_data/TvalPCA_0%i.txt", i);

		ReadEigenFaceData(maleValidationFaces, femaleValidationFaces, infname, tgfname);

		for (int i = 0; i < maleValidationFaces.size(); i++)
		{
			count++;
			float d1 = discriminant_case_3(maleValidationFaces[i], mm, mc, 0.5);
			float d2 = discriminant_case_3(maleValidationFaces[i], fm, fc, 0.5);

			//cout << "face: " << maleTrainingFaces[i] << endl;
			//cout << "d1: " << d1 << endl;
			//cout << "d2: " << d2 << endl;

			if (d1 >= d2)
			{
				maleCorrect++;
			}
		}

		for (int i = 0; i < femaleValidationFaces.size(); i++)
		{
			count++;
			float d1 = discriminant_case_3(femaleValidationFaces[i], fm, fc, 0.5);
			float d2 = discriminant_case_3(femaleValidationFaces[i], mm, mc, 0.5);

			if (d1 >= d2)
			{
				femaleCorrect++;
			}
		}

		cout << "number of faces tested: " << count << endl;
		cout << "male correct: " << maleCorrect << endl;
		cout << "female correct: " << femaleCorrect << endl;
		cout << "total correct: " << maleCorrect + femaleCorrect << endl;
		cout << "percentage: " << (float)(maleCorrect + femaleCorrect)/(float)count << endl;
		cout << endl;

		string outfname("48_60_bayes_results_0");
		outfname = outfname + to_string(i);
		outfname = outfname + ".txt";
		ofstream f(outfname);

		if (f.is_open())
		{
			f << "number of faces tested: " << count << endl;
			f << "male correct: " << maleCorrect << endl;
			f << "female correct: " << femaleCorrect << endl;
			f << "total correct: " << maleCorrect + femaleCorrect << endl;
			f << "percentage: " << (float)(maleCorrect + femaleCorrect)/(float)count << endl;
			f.close();
		}
	}
}


float discriminant_min_distance(VectorXf x, VectorXf mu)
{
	return -1.0 * squared(x-mu);
}



bool threshold_case_3(VectorXf x, VectorXf mu, MatrixXf sigma, float threshold)
{
	return (-0.5 * (x - mu).transpose() * sigma.inverse() * (x - mu)) > threshold;
}
