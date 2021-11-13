#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace Eigen;
using namespace std;

int TRAINING = 1;
int TESTING = 2;

void generate_data(char* filename, int datatype, string imagetype);

int main()
{
	char filename[256];

	// reformat data for all training, test, and validation files
	for(int i=1;i<=3;i++)
	{
		sprintf(filename, "trPCA_0%i", i);
		generate_data(filename, TRAINING, "16_20");
		sprintf(filename, "tsPCA_0%i", i);
		generate_data(filename, TESTING, "16_20");
		sprintf(filename, "valPCA_0%i", i);
		generate_data(filename, TESTING, "16_20");

		sprintf(filename, "trPCA_0%i", i);
		generate_data(filename, TRAINING, "48_60");
		sprintf(filename, "tsPCA_0%i", i);
		generate_data(filename, TESTING, "48_60");
		sprintf(filename, "valPCA_0%i", i);
		generate_data(filename, TESTING, "48_60");
	}
}

void generate_data(char* filename, int datatype, string imagetype)
{
	// get input and output file streams
	string inFile = "./genderdata/";
	inFile += imagetype;
	inFile += "/";
	inFile += imagetype;
	inFile += "_reformatted_data/";
	string curData = "";
	char curLabel;
	ifstream finData;
	ifstream finLabel;
	istringstream iss;
	string output_name = "./New_SVM_Data/";
	output_name += imagetype;
	output_name += "-";
	output_name += filename;
	output_name += "-new.txt";
	ofstream file_output(output_name);

	float curFloat = 0.0;
	int counter = 1;
	inFile += filename;
	inFile += "-new.txt";
	finData.open(inFile);
	inFile = "./genderdata/";
	inFile += imagetype;
	inFile += "/T";
	inFile += filename;
	inFile += ".txt";
	finLabel.open(inFile);

	char temp;

	// combine the labels and eigen face representations
	// for each line:
	// <label> <eigen face representation>
	// first 30 for training
	if(datatype == TRAINING)
	{
		int eigen_index = 0;
		while(eigen_index < 30 && getline(finData, curData) && finLabel.get(curLabel))
		{
			while(curLabel == ' ')
			{
				finLabel.get(curLabel);
			}
			file_output << curLabel;
			iss.clear();
			iss.str(curData);
			counter = 1;
			while(iss >> curFloat)
			{
				file_output << " " << counter << ":" << curFloat;
				counter++;
			}

			file_output << endl;
			eigen_index++;
		}
	}
	// all for test and validation
	else if(datatype == TESTING)
	{
		while(getline(finData, curData) && finLabel.get(curLabel))
		{
			while(curLabel == ' ')
			{
				finLabel.get(curLabel);
			}

			if(curLabel != '1' && curLabel != '2')
			{
				break;
			}
/*
			if(curLabel == '\0')
			{
				break;
			}*/

			file_output << curLabel;
			iss.clear();
			iss.str(curData);
			counter = 1;
			while(iss >> curFloat)
			{
				file_output << " " << counter << ":" << curFloat;
				counter++;
			}
			file_output << endl;
		}
	}

	// close stream
	finData.close();
	finLabel.close();
	file_output.close();
}
