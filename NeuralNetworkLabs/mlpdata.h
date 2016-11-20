// Header file for mlpdata 
// has class for storing data set for mlp
// inputs and targets can be read from file or from an array
// space also for calculated outputs and sum squares errors
// 2012, now have data fiels for logic; numerical and classification
//
// Dr Richard Mitchell 15/12/06 ...7/6/11 .. 23/08/11 .. 07/09/12 .. 18/09/15

#include <vector>
using namespace std;

class dataset {
	int numdataset;
	int numinputs;
	int numoutputs;
	int numinrow;		// is numinputs + 2 * numoutputs
	int datatype;		// 0 for logic, 1 for numerical, 2 for classifier
	vector<double> alldata;	// array for inputs, outputs and targets
	vector<double> minNData;	// array for minimum values of each input/target
	vector<double> maxNData;	// array for maximum values of each input/target
	vector<double> errors;		// array for errors of each output : used fro errors, SSE, etc
	vector<double> classifications;  // array for % of correct classifications
	vector<double> scaleddata;	// for rescaling outputs at end for display
	char dataname[40];		// name of data
	void GetMemory(char *name);
	void ScaleInsTargets(void);
	double DeScale(double val, int sCt, int forOut);		
		// descale value val, according to datatype, min and max values index sCt, forOut is true if for target or out
public:
	dataset();
	dataset(char *filename, char *name);
	dataset(int nin, int nout, int nset, vector<double> netdata, char *name);
	~dataset();
	vector<double> GetNthInputs (int n);
		// return vector of inputs of nth item in data set
	vector<double> GetNthTargets (int n);
		// return vector of targets of nth item in data set
	vector<double> GetNthOutputs (int n);
		// return vector of array of output of nth item in data set
	vector<double> GetNthErrors (int n);
		// return vector of errors of nth item in data set (targets - outputs)
	double GetNthError(int n);
		// return the error of the nth item in the data set (target - output
	void SetNthOutputs(int n, vector<double> outputs);
		// copy actual calculated outputs into nth item in data set
	void SetNthOutput(int n, double oneout);
		// for network with one output ... store oneout in dataset
	vector<double> CalcSSE (void);
		// calculate SSE across data set, and return address of array with SSEs for each output
	double TotalSSE (void);
		// calc and return sum of all SSEs of data in set
	vector<double> CalcCorrectClassifications(void);
		// calculate and return vector of % of correct classifications
	vector<double> CalcScaledData (int n, char which);
		// calculate and return vector of scaled version for nth output set
	int numIns (void);
		// return number of inputs
	int numOuts (void);
		// return number of outputs
	int numData (void);
		// return number of data sets
	void printarray (char *s, char which, int n, int nl = 0);
		// print s then specifc array and \n if nl
		// if which is 'I' print inputs; if 'O' print outputs; 
		//          if 'T print targets, if 'S' print SSEs
		// n specifies nth set of inputs,outputs,targets
	void printdata (int showall);
		// print data in set (all if showall) then its SSE
	void savedata (int goplot = 0);
		// save data set (ins, targets and outs) into file
	    // if goplot, then call tadpole program to plot
};



