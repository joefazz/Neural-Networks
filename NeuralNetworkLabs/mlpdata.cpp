
// Library file for handling datasets for MLPs in CS2NN16
// Prof Richard Mitchell This version 29/09/16

using namespace std;

#include <fstream>
#include <vector>
#include "mlpdata.h"
#include <iostream>
#include <iomanip>



void arrout (char *s, int num, vector<double> data, int nl) {
	// routine to output the num values in array data
	// s is a string which precedes the array
	// if nl is true, then \n is then output
	cout.precision (3);				// set precision : format of numbers
	cout << s;						// output s
									// next do num doubles in data
	for (int ct = 0; ct < num; ct++) cout << setw(8) << data[ct];
	if (nl) cout << "\n";			// if desired output newline
}

// Implementation of dataset class

dataset::dataset() {
	// argument less constructor, just initialises to 0
	GetMemory("");   // initialise all relevant memory to 0
}

dataset::dataset (char *filename, char *name) {
	// constructor where argument is name of file which contains data
	// this opens files, initialises the number of inputs, etc
	// creates space for the data
	// then reads all the data from the file
	ifstream datafile;				// define stream variable for data

	datafile.open(filename);		// open file of given name
	
	if (datafile.is_open()) {
	datafile >> numinputs >> numoutputs >> numdataset >> datatype;
									// read amounts of data from file and type
	GetMemory(name);				// create space for in/outs/errors etc
	int nd, ndi=0, ct;				// counters

	if (datatype > 0)	{		// if not logic, then read min/max for ins/targets
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read min values inputs and targets
			datafile >> minNData[ct];
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read max inputs and targets
			datafile >> maxNData[ct];
		for (ct=0; ct<numoutputs; ct++) {				// ensure outputs have same min/max as targets
			minNData[numinputs+numoutputs+ct] = minNData[numinputs+ct];
			maxNData[numinputs+numoutputs+ct] = maxNData[numinputs+ct];
		}
	}

	for (nd=0; nd<numdataset; nd++) {	// read each item from set
	
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read n inputs and targets
			datafile >> alldata[ndi++];
	
		ndi += numoutputs;								// skip passed actual outputs
	}
	ScaleInsTargets();					// scaled inputs and targets 
	datafile.close();					// close file
	}
	else GetMemory("");
}

dataset::dataset (int nin, int nout, int nset, vector<double> netdata, char *name) {
	// constructor to create dataset where raw data in array 
	// arguments passed numbers of inputs, nin, outputs, nout, and in set, nset
	// data is a large enough array
	numinputs = nin;				// first store data sizes
	numoutputs = nout;
	numdataset = nset;									
	datatype = 0;
	GetMemory(name);					// create space for in/outs/errors etc

	int wd = 0, nd, ndi=0, ct;

	for (nd=0; nd<numdataset; nd++) {	// now read each data item from data
		
		for (ct=0; ct<numinputs + numoutputs; ct++)		// read n inputs and targets
			alldata[ndi++] = netdata[wd++];
	
		ndi += numoutputs;								// skip passed actual outputs
	}
}

dataset::~dataset () {
				// return memory to heap ... not needed as vectors used
}

void dataset::GetMemory(char *name) {
		// create space for vectors for inputs, outputs, targets and SSEs
						// first initialise memory
	numinrow = numinputs + 2 * numoutputs;		// how many elements in a row ie inputs, targets, outputs
	alldata.resize(numinrow * numdataset);		// get memory for all data
	errors.resize(numoutputs);					// and for vector of SSEs
	classifications.resize(numoutputs);			// and for % classifications
	scaleddata.resize(numinrow);				// and for row of re-Scaled data
	minNData.resize(numinrow);					// for min of all ins/targets/outputs
	maxNData.resize(numinrow);					// for max of all ins/targets/outputs
	for (int ct=0; ct<numinrow; ct++)	{		// set min/max to 0 so no scaling
		minNData[ct] = 0;
		maxNData[ct] = 0;
	}
	strcpy_s(dataname, 40, name);				// copy name to dataname
}

void dataset::ScaleInsTargets(void) {
		// scale all ins/targets linearly to 0.1..0.9 
		// minNData and maxNData have minimum and maximum values for each in/target
	int ndi = 0;						// index into alldata.
	for (int nd=0; nd<numdataset; nd++) {	// read each item from set
	
		for (int ct=0; ct<numinputs + numoutputs; ct++)	{	// read n inputs and targets
			if (maxNData[ct] > minNData[ct])
				alldata[ndi] = 0.1 + 0.8 * (alldata[ndi] - minNData[ct]) / (maxNData[ct] - minNData[ct]);
			ndi++;
		}
		ndi += numoutputs;								// skip passed actual outputs
	}

}

vector<double> dataset::GetNthInputs (int n) {
		// return a vector the inputs of nth item in data set
	vector <double> ans;
	for (int ct = 0; ct < numinputs; ct++) ans.push_back(alldata[n*numinrow + ct]);
	return ans;
}

vector<double> dataset::GetNthTargets (int n){
		// return a vector of targets of nth item in data set
	vector <double> ans;
	for (int ct = 0; ct < numoutputs; ct++) ans.push_back(alldata[n*numinrow + numinputs + ct]);
	return ans;
}

vector<double> dataset::GetNthOutputs (int n){
		// return address of (first) output of nth item in data set
	vector <double> ans;
	for (int ct = 0; ct < numoutputs; ct++) ans.push_back(alldata[n*numinrow + numinputs + numoutputs + ct]);
	return ans;
}

void dataset::SetNthOutputs(int n, vector<double> outputs) {
		// copy actual calculated outputs into nth item in data set
	for (int ct = 0; ct < numoutputs; ct++)
		alldata[n*numinrow + numinputs + numoutputs + ct] = outputs[ct];
}

void dataset::SetNthOutput(int n, double oneout) {
	// for network with one output ... store oneout in dataset
	alldata[n*numinrow + numinputs + numoutputs] = oneout;
}

vector<double> dataset::GetNthErrors (int n){
		// calculate and return errors (targets-outouts) for n'th set in dataset
		// for each output

  int tarNdx = n*numinrow + numinputs;		// index into targets array
  int opNdx = tarNdx+numoutputs;			// and to outputs
 
  for (int ct = 0; ct < numoutputs; ct++)
	  errors[ct] = alldata[tarNdx++] - alldata[opNdx++];	// calc next error
  return errors;
}

double dataset::GetNthError(int n) {
	// return the error of the nth item in the data set (target - output)
	return alldata[n*numinrow + numinputs] - alldata[n*numinrow + numinputs + 1];
}

double sqr (double v) { return v*v; }		// calculate v^2

vector <double> dataset::CalcSSE (void){
		// calculate and return sum of square of errors (targets-outouts)
		// for each output
  int ct, nct; 
  int tarNdx;	// index into targets array
  int opNdx;	// and to outputs

  for (ct = 0; ct < numoutputs; ct++)
	  errors[ct] = 0;
	  
	for (nct=0; nct<numdataset; nct++) {
		tarNdx = nct*numinrow + numinputs;		// index to first target for nct'th item
		opNdx = tarNdx + numoutputs;			// and to first output
		for (ct=0; ct<numoutputs; ct++) 
			errors[ct] += sqr(alldata[tarNdx++] - alldata[opNdx++]);	// add error ^ 2, and move indexes on
	}
    for (ct=0; ct<numoutputs; ct++) 
		errors[ct] = errors[ct] / numdataset;			// divide by num in set
	return errors;
}

vector<double> dataset:: CalcCorrectClassifications(void) {
		// calculate and return sum of square of errors (targets-outouts)
		// for each output
	int tarNdx;	// index into targets array
	int outNdx; // into output array

	int ct, nct;

    for (ct=0; ct<numoutputs; ct++) 
		classifications [ct] = 0;							// correct classifications = 0
	for (nct=0; nct<numdataset; nct++) {
		tarNdx = nct*numinrow + numinputs;					// have index of first target for nct'th item in set
		outNdx = tarNdx + numoutputs;						// and of first output
		for (ct=0; ct<numoutputs; ct++)
			if (abs (DeScale(alldata[tarNdx++], ct, 1) - DeScale(alldata[outNdx++], ct, 1)) < 0.01) classifications[ct] += 1;	// if scaled Target ~ Scaled Output
	}
    for (ct=0; ct<numoutputs; ct++) 
		classifications[ct] = (100 * classifications[ct] ) / numdataset;			// turn result into a %
	return classifications;
}

double dataset::DeScale(double val, int sCt, int forOut) {
	// descale value val, according to datatype, min and max values (which are at index [sCt]
	// forOut is true if this applies to target / output - so for instance floor/ceil to 0 or 1
	double ans = val;
	if (datatype == 0) {
		if (forOut) {
			if (val <= 0.5) ans = 0; else ans = 1;
		}
	}
	else {
		if (maxNData[sCt] > minNData[sCt])
			ans = minNData[sCt] + (val - 0.1) * (maxNData[sCt] - minNData[sCt]) / 0.8;
		if ((datatype == 2) && forOut)
			ans = floor(0.5 + val);
	}
	return ans;
}

vector <double> dataset::CalcScaledData(int n, char which) {
int dataNdx = n*numinrow;
int minnum, maxnum;
	switch (which) {
	   case 'I' :  minnum = 0; maxnum = numinputs; break;
	   case 'T' :  minnum = numinputs; maxnum = minnum + numoutputs; break;
	   case 'O' :  minnum = numinputs+numoutputs; maxnum = minnum + numoutputs; break;
	   case 'A' :  minnum = 0; maxnum = numinrow; break;
	} 
	for (int ct=minnum; ct<maxnum; ct++) {
		scaleddata[ct - minnum] = DeScale(alldata[ct + dataNdx], ct, ct>=numinputs);
	}		// call descale for each item
	return scaleddata;
}

double dataset::TotalSSE (void) {
		// calc and return sum of all SSEs of data in set
	double ans = 0;
	CalcSSE();
	for (int ct=0; ct<numoutputs; ct++) ans += errors[ct];
	return ans;
}

int dataset::numIns(void) {
		// return number of data sets
	return numinputs;
}

int dataset::numOuts(void) {
		// return number of data sets
	return numoutputs;
}

int dataset::numData(void) {
		// return number of data sets
	return numdataset;
}

void dataset::printarray (char *s, char which, int n, int nl) {
		// print s then specifc array and \n if nl
		// if which is 'I' print inputs; if 'O' print outputs; 
		//          if 'T print targets, if 'S' print SSEs
		// n specifies nth set of inputs,outputs, targets
	switch (which) {
	case 'i': 
	case 'I' : arrout(s, numinputs, GetNthInputs(n), nl); break;
	case 'o' :
	case 'O' : arrout(s, numoutputs, GetNthOutputs(n), nl); break;
	case 't' :
	case 'T' : arrout(s, numoutputs, GetNthTargets(n), nl); break;
	case 's' :
	case 'S' : arrout(s, numoutputs, CalcSSE(), nl); break;
	case 'c' :
	case 'C' : arrout(s, numoutputs, CalcCorrectClassifications(), nl); break;
	case 'r' :
	case 'R' : arrout(s, numoutputs, CalcScaledData(n, 'O'), nl); break;
	}
}

void dataset::printdata (int showall) {
	// pass a training set in data to network, show results
	if ( (showall > 0) && (showall < 3) ) {
		cout << setw(1 + 8*numinputs) << "Inputs" 
			 << setw(3 + 8*numoutputs) << "Targets"
			 << setw(3 + 8*numoutputs) << " Actuals" 
			 << setw(3 + 8*numoutputs) << "Rescaled\n";
	  for (int ct=0; ct<numdataset; ct++) {
		printarray (" ", 'I', ct, 0);
		printarray (" : ", 'T', ct, 0);
		printarray (" : ", 'O', ct, 0);
		printarray (" : ", 'R', ct, 0);
		cout << "\n";
	  }
	}
	else  cout << dataname << " : ";
	if (showall >= 3) printarray("% Correct Classifications ", 'C', 0, 1);
	else {
	  printarray ("Mean Sum Square Errors are ", 'S', 0, 1);
	  if (showall && (datatype != 1) ) printarray("% Correct Classifications ", 'C', 0, 1);
	}

}

void dataset::savedata (int goplot) {
		// save data set (ins, targets and outs)
	ofstream datafile;				// define stream variable for data
	int ct2;
	char temp[80];
	strcpy_s(temp, 50, dataname);		// copy name to temp
	strcat_s(temp, 60, "full.txt");		// add full.txt to name
	datafile.open(temp);				// open file of given name for writing to
	if (datafile.is_open())   {
      datafile << numinputs << " " << numoutputs << " " << numdataset << "\n";
      for (int ct=0; ct<numdataset; ct++) {
	    CalcScaledData(ct, 'A');
		for (ct2=0; ct2<numinrow; ct2++) datafile << scaleddata[ct2] << "\t";
		datafile << "\n";
	  } 
	  datafile.close();
	  if (goplot) {		// evoke tadpole.exe with name of file  .. does not work automatically
//		  strcpy (temp, "tadpole.exe ");
//		  strcat (temp, dataname);
//		  strcat(temp, "full.txt");
//		  system (temp);
		  cout << "Invoke the tadpole program, select file " << dataname << "full.txt and plot response.\n" ;
	  }
	}
	else cout << "Unable to create " << temp << "\n";
}

