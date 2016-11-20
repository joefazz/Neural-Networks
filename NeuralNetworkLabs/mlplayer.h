// include file for single and multiple layer network
// Dr Richard Mitchell 15/12/06 ... 18/09/13
// Adapted by

// First include definition of dataset class

#include "mlpdata.h"

// Next define classes for layers of networks

class LinearLayerNetwork {					// simple layer with linear activation
protected:
   friend class MultiLayerNetwork;				// so MultiLayerNetwork can access protected functions
   int numInputs, numNeurons, numWeights;		// how many inputs, neurons and weights
   vector<double> outputs;						// array of neuron Outputs
   vector<double> deltas;						// and Deltas 
   vector<double> weights;						// array of weights
   vector<double> changeInWeights;				// array of weight changes

   
   virtual void CalcOutputs (vector<double> ins);
		// ins are passed to net, weighted sums of ins are calculated and stored in outputs
   virtual void SendOutputs (int n, dataset &data);
		// send calculated network outputs for storing in the nth outputs in data set data
   virtual void FindDeltas (vector<double> errors);
        // find the deltas: from errors between targets and outputs
   virtual void ChangeAllWeights (vector<double> ins, double learnRate, double momentum);
		// change all weights in layer using deltas, inputs (ins), learning rate and momentum
   virtual vector <double> WeightedDeltas();
		// find weighted sum of deltas in this layer being errors in previous layer
   virtual vector <double> PrevLayerErrors();
  

public:
   LinearLayerNetwork (int numIns, int numOuts);
		// constructor pass num of inputs and outputs (=neurons)
   virtual ~LinearLayerNetwork ();
		// destructor
   virtual void ComputeNetwork (dataset &data);
		// pass whole dataset to network, calculating outputs, storing in data
   virtual void AdaptNetwork (dataset &data, double learnRate, double momentum);
		// pass whole dataset to network : for each item
		//   calculate outputs, copying them back to data
		//   adjust weights using the delta rule : targets are in data
		//     adapt using learning rate and momentum
   virtual void SetTheWeights (vector<double> initWt);
		// initialise weights in network to values in initWt
   virtual int HowManyWeights (void);
	    // return number of weights
   virtual vector <double> ReturnTheWeights ();
		// return the weights in the network into theWts

};


class SigmoidalLayerNetwork : public LinearLayerNetwork {
protected:		// Output Layer with Sigmoid Activation
   virtual void FindDeltas (vector<double> errors);
        // find the deltas: being errors * output * (1 - output)
   virtual void CalcOutputs (vector<double> ins);
		// Calculate outputs as Sigmoid(Weighted Sum of ins)
public:
   SigmoidalLayerNetwork (int numIns, int numOuts); 
		// constructor
   virtual ~SigmoidalLayerNetwork ();			
		// destructor
};


class MultiLayerNetwork : public SigmoidalLayerNetwork {
				// Network : a Sigmoid Activated Hidden Layer with output layer
protected:
   LinearLayerNetwork *nextlayer;					// pointer to next layer
   virtual void CalcOutputs (vector<double> ins);
		// Calculate outputs of whole network from inputs ins 
   virtual void SendOutputs (int n, dataset &data);
		// send outputs of final layer for storing in the nth outputs in data set data
   virtual void FindDeltas (vector<double> errors);
        // find the deltas in next and this layer
   virtual void ChangeAllWeights (vector<double> ins, double learnRate, double momentum);
		// calc change in weights using deltas, inputs, learning rate and mmtum
public:
   MultiLayerNetwork (int numIns, int numOuts, LinearLayerNetwork *tonextlayer); 
		// constructor
   virtual ~MultiLayerNetwork ();			
		// destructor
   virtual void SetTheWeights (vector<double> initWt);
		// set the woeights of main layer and the nextlayer(s) using values in initWt
   virtual int HowManyWeights (void);
	    // return number of weights in whole network
   virtual vector<double> ReturnTheWeights ();
		// return the weights of whole network into theWts
};


