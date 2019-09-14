package predictors;

import java.util.ArrayList;

import dataprocessing.*;


public class MLP
{

public static java.util.Random generator = new java.util.Random();

///////////////////////////////////////////////////////////////////////
    
public double learningRate   = 0.1;
public int currentIteration  = 0;
public boolean sigmoidOutput = false;

protected double outputNode, outputLayerBias;
protected double hiddenNodes[], hiddenLayerBias[];
protected double hiddenLayerWeights[][], outputLayerWeights[];

public static final int LINEAR  = 0;
public static final int SIGMOID = 1;

/////////////////////////////////////////////////////////////////////////

/**
 * Constructor for the MLP network.
 * 
 * @param numberOfInputs The number of inputs this MLP should have
 * @param numberOfHiddenNodes The number of hidden nodes per layer that this MLP should have. Sorry, but limited to only one hidden layer at present.
 * 
 */
public MLP ( int numberOfInputs, int numberOfHiddenNodes, int outputType )
{
	outputNode			= 0;
	outputLayerBias		= 0;
	outputLayerWeights	= new double[numberOfHiddenNodes];

	if (outputType == MLP.SIGMOID) sigmoidOutput = true;
	
	outputLayerBias = generator.nextDouble()-0.5;
	for(int h=0; h<numberOfHiddenNodes; h++)
	{
		outputLayerWeights[h] = generator.nextDouble()-0.5;
	}		
	
	hiddenNodes		= new double[numberOfHiddenNodes];
	hiddenLayerBias		= new double[numberOfHiddenNodes];
	hiddenLayerWeights	= new double[numberOfHiddenNodes][];
	for(int h=0; h<numberOfHiddenNodes; h++)
	{
		hiddenLayerBias[h] = generator.nextDouble()-0.5;
		hiddenLayerWeights[h] = new double[numberOfInputs];

		for(int i=0; i<numberOfInputs; i++)
		{
			hiddenLayerWeights[h][i] = generator.nextDouble()-0.5;
		}
	}
}

/////////////////////////////////////////////////////////////////////////

/**
 * Get the classification from this network, returns an integer, either 0 or 1. 
 */
public int getClassification()
{
	return (int)Math.round(outputNode);
}

/////////////////////////////////////////////////////////////////////////

/**
 * Get the output from this network, real valued, if this network has a sigmoid output, then this method
 * will return the posterior probability of class membership.
 */
public double getOutput()
{
	return outputNode;
}

/////////////////////////////////////////////////////////////////////////

/**
 * Forward pass for the MLP network.
 * 
 * @param x An array of type double, containing inputs to be passed through this network.
 */
public void forwardPass(double x[])
{
	//SET OUTPUT TO ZERO
 	//
	outputNode = 0;
	
	//FOR EACH HIDDEN NODE
	//
	for(int h=0; h<hiddenNodes.length; h++)
	{
		//SET TO ZERO
	  	//
		hiddenNodes[h]=0;

		//CALCULATE ITS WEIGHTED INPUTS
		//
		for(int i=0; i<x.length; i++)
		{
			hiddenNodes[h] += hiddenLayerWeights[h][i]*x[i];
		}
		
		//TAKE ACCOUNT OF THIS NODE'S BIAS (BEING LEARNT AS A WEIGHT)
		//
		hiddenNodes[h] += hiddenLayerBias[h]*-1;
		
		//APPLY THE SIGMOID ACTIVATION FUNCTION
		//
		hiddenNodes[h] = 1.0/(1.0+Math.exp(-hiddenNodes[h]));
		
		//PASS ITS VALUE FORWARD TO THE OUTPUT LAYER NODE
		//
		outputNode += hiddenNodes[h]*outputLayerWeights[h];
			
		//THEN LOOP BACK TO THE TOP AND TAKE THE NEXT HIDDEN NODE
		//...	
	}
	
	//TAKE ACCOUNT OF THE OUTPUT NODE'S BIAS (BEING LEARNT AS A WEIGHT)
	//
	outputNode += outputLayerBias*-1;

	
	//AND APPLY THE SIGMOID ACTIVATION FUNCTION IF NECESSARY
	//
	if (sigmoidOutput)
	{
		outputNode = 1.0/(1.0+Math.exp(-outputNode));
	}
}

/////////////////////////////////////////////////////////////////////////

/**
 * Backward pass for the MLP network.
 * 
 * @param x An array of type double, containing inputs to be passed through this network.
 * @param target Target value corresponding to the inputs.
 */
public void backwardPass(double x[], double target) {
	backwardPass( x, target, 0 );
}

/**
 * Backward pass for the MLP network.
 * 
 * @param x An array of type double, containing inputs to be passed through this network.
 * @param target Target value corresponding to the inputs.
 * @param penalty A penalty to apply to the error function.
 */
public void backwardPass(double x[], double target, double penalty)
{
	//COMPUTE THE EFFECT THAT THE OUTPUT NODE HAD ON THE ERROR FUNCTION
	//
	double outputDelta = (outputNode-target) + penalty;
	
	if (sigmoidOutput)
	{
		outputDelta = outputDelta*(outputNode*(1-outputNode));
	}

	//FOR EACH HIDDEN NODE
	//	
	for(int h=0; h<hiddenNodes.length; h++)
	{
		//COMPUTE THE EFFECT THAT THIS HIDDEN NODE HAD ON THE OUTPUT NODE ERROR
		//
		double hiddenDelta = outputDelta*outputLayerWeights[h]*(hiddenNodes[h]*(1-hiddenNodes[h]));

		//FOR EACH INPUT THAT LEADS INTO THIS HIDDEN NODE
		//
		for(int i=0; i<x.length; i++)
		{
			//UPDATE THE WEIGHT BETWEEN THAT INPUT NODE AND THIS HIDDEN NODE
			//
			hiddenLayerWeights[h][i] += -learningRate*hiddenDelta*x[i];
		}
		
		//UPDATE THE BIAS FOR THIS HIDDEN NODE
		//
		hiddenLayerBias[h]	+= -learningRate*hiddenDelta*-1;
		
		//UPDATE THE WEIGHTS BETWEEN THIS HIDDEN NODE THE OUTPUT NODE
		//
		outputLayerWeights[h]	+= -learningRate*outputDelta*hiddenNodes[h];
	}
	
	//UPDATE THE BIASES FOR THE OUTPUT NODES
	//
	outputLayerBias += -learningRate*outputDelta*-1;
}

/////////////////////////////////////////////////////////////////////////

public void train(ArrayList mylist, int howManyIterations)
{
    int finalIteration = currentIteration + howManyIterations;
    double fbar = 0;

    for (int i = currentIteration; i < finalIteration; i++)
    {
        for (int pat = 0; pat < mylist.size(); pat++)
        {
            Example e = (Example) mylist.get(pat);
            
            this.backwardPass(e.inputs, e.target);        
        }
    }
    currentIteration += howManyIterations;
}    

/////////////////////////////////////////////////////////////////////////

public double test(ArrayList mylist)
{
	double error=0;
    for (int pat = 0; pat < mylist.size(); pat++)
    {
        Example e = (Example) mylist.get(pat);   
        this.forwardPass(e.inputs);     
            
        if (e.target != this.getClassification()) error++;
    }
    return error / mylist.size();
}    

/////////////////////////////////////////////////////////////////////////

}
