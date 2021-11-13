#include <iostream>
#include <math.h>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include "Bayes.cpp"
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

void readFaces(vector<VectorXf> &maleFaces, vector<VectorXf> &femaleFaces, string examplePath, string targetPath)
{
	// clear vectors
	maleFaces.clear();
	femaleFaces.clear();

	// open file streams
	ifstream faceFile(examplePath);
	ifstream targetFile(targetPath);

	// get data from files
	// store data in vectors
	string line;
	getline(targetFile, line);
	istringstream targetStream(line);

	while (getline(faceFile, line))
	{
		istringstream iss(line);

		vector<float> values;
	
		float val = 0;

		while(true)
		{
			if (! (iss >> val) )
			{
				break;
			}
			values.push_back(val);
		}

		VectorXf newFace(values.size());

		for(unsigned int i = 0; i < values.size(); i++)
		{
			newFace.row(i) << values[i];
		}

		int bayesClass = 0;

		if (! (targetStream >> bayesClass) )
		{
			cout << "Unable to read targets" << endl;
			break;
		}
		if (bayesClass == 1)
		{
			maleFaces.push_back(newFace);
		}
		else
		{
			femaleFaces.push_back(newFace);
		}
	}
}

int main(){

	//read in training faces

	vector<VectorXf> maleTrainingFaces;
	vector<VectorXf> femaleTrainingFaces;

	readFaces(maleTrainingFaces, femaleTrainingFaces, "genderdata/48_60/trPCA_01.txt", "genderdata/48_60/TtrPCA_01.txt");

	//calculate mean and covariance using ML

	VectorXf maleMean = ml_mean(maleTrainingFaces);
	VectorXf femaleMean = ml_mean(femaleTrainingFaces);

	MatrixXf maleCovariance = ml_covariance(maleTrainingFaces, maleMean);
	MatrixXf femaleCovariance = ml_covariance(femaleTrainingFaces, femaleMean);

	//read in test faces

	vector<VectorXf> maleTestFaces;
	vector<VectorXf> femaleTestFaces;

	readFaces(maleTestFaces, femaleTestFaces, "genderdata/48_60/tsPCA_01.txt", "genderdata/48_60/TtsPCA_01.txt");


	//count the number of correct classifications
	int count = 0;

	//male test faces
	for (int i = 0; i < maleTestFaces.size(); i++)
	{
		int classify = classify_case_3(maleTestFaces[i], maleMean, femaleMean, maleCovariance, femaleCovariance);
		if (classify == 1)
		{
			count++;
		}
	}

	//female test faces
	for (int i = 0; i < femaleTestFaces.size(); i++)
	{
		int classify = classify_case_3(femaleTestFaces[i], maleMean, femaleMean, maleCovariance, femaleCovariance);
		if (classify == 2)
		{
			count++;
		}
	}




	return 0;

}
