package predictors;

import java.util.ArrayList;

import dataprocessing.*;

public class Ensemble
{

/**
 * ArrayList containing all networks in this ensemble.
 * Add to this list using the <code>add( MLP )</code> method.
 */
protected  ArrayList<MLP> networks = new ArrayList<MLP>();

/**
 * Number of networks in this ensemble.
 */
public  int numNets;

/**
 * Total number of iterations trained on this ensemble so far.
 */
public  int currentIteration=0;

/**
 * True if this ensemble is doing a classification problem, false otherwise.
 */
public boolean classificationEnsemble=false;

/**
 * True if this ensemble is combined by a majority vote.
 */
public boolean voting=false;

/**
 * Strength parameter used in NC learning, upper bound of 1.0
 * Alternative is Gamma, upper bound M/(2M-2)
 */
public double lambda=0.0;

///////////////////////////////////////////////////////////////


public void train(ArrayList mylist, int howManyIterations)
{
    int finalIteration = currentIteration + howManyIterations;
    double fbar = 0;

    for (int i = currentIteration; i < finalIteration; i++)
    {
        for (int pat = 0; pat < mylist.size(); pat++)
        {
            Example e = (Example) mylist.get(pat);
            fbar = average( e.inputs );
            
            MLP x = null;
            double newTarget=0;
            for (int n = 0; n < numNets; n++)
            {
                x = (MLP) networks.get(n);
               
                //NC METHOD USING PENALTY TERM
                //
                double penalty = -lambda * (x.outputNode - fbar);
                x.backwardPass(e.inputs, e.target, penalty);
            }

        }//endfor p
    }//endfor i

    currentIteration += howManyIterations;
}

/////////////////////////////////////////////////////////////////////////

/**
 * Get the output of this ensemble for the provided input vector.
 * 
 * @param inputs An array of input attributes to pass through the ensemble.
 * @return The classification of this input vector, in the range 0 to <code>numOutputs-1</code>
 * The combination method of the ensemble will depend on whether the <code>voting</code> flag is set.
 */
public double ensembleOutput( double[] inputs )
{
	double out = 0.0;

	if (voting)
	{
		out = majorityVote(inputs);
	}
	else
	{
		out = average(inputs);
		if (classificationEnsemble) out = Math.round(out);
	}
	
	return out;
}

///////////////////////////////////////////////////////////////

protected double average( double[] inputs )
{
	double output=0.0;
	MLP x = null;
	
	for (int n=0; n<numNets; n++)
	{	
		x = (MLP)networks.get(n);
		x.forwardPass( inputs );

		output += x.outputNode;
	}

	return output/(double)numNets;
}

///////////////////////////////////////////////////////////////

/**
 * @param mylist A dataset which to evaluate our ensemble on, i.e. an ArrayList of <code>Example</code> objects.
 * @return The error of this ensemble on the supplied dataset.<br>
 * 
 * If <code>classifier=TRUE</code>, this returns the error rate.<br>
 * If <code>classifier=FALSE</code>, this returns the mean squared error.
 * 
 */
public double test(ArrayList mylist)
{
	if (classificationEnsemble)
		return errorRate(mylist);
	else
		return mse(mylist);
}

///////////////////////////////////////////////////////////////

protected double majorityVote( double[] inputs )
{
	MLP x = null;
	int[] votes = new int[2];
	
	
	for (int n=0; n<numNets; n++)
	{
		x = (MLP)networks.get(n);
		x.forwardPass( inputs );
		
		votes[ x.getClassification() ]++;
	}

	int max=0, output=0;
	for (int c=0; c<votes.length; c++)
	{
		if (votes[c] > max)
		{
			max=votes[c];
			output = c;
		}
	}

	return output;
}

///////////////////////////////////////////////////////////////

protected double mse(ArrayList mylist)
{
	double mse=0, output=0;

	for(int p=0; p<mylist.size(); p++)
	{
		Example thisExample = (Example)mylist.get(p);

		output = ensembleOutput( thisExample.inputs );

		mse += (thisExample.target - output)*(thisExample.target - output);
	}

	return (mse/(double)mylist.size());
}

///////////////////////////////////////////////////////////////

protected double errorRate(ArrayList mylist)
{
	int errors=0;

	for(int p=0; p<mylist.size(); p++)
	{
		Example thisExample = (Example)mylist.get(p);
		
		int guess = (int) ensembleOutput( thisExample.inputs );
		
		int answer = (int)Math.round(thisExample.target);
		
		if (guess != answer) 
		{
			errors++;
		}
	}

	return (errors/(double)mylist.size());
}

///////////////////////////////////////////////////////////////

/**
 * Add a network to this ensemble.
 * 
 * @param x - A network to add.
 */
public void add( MLP x )
{
	networks.add( x );
	numNets++;
}

////////////////////////////////////////////////////////////////////////////////////////

/**
 * Sets the learning rate for this ensemble.
 * 
 * @param alpha - A learning rate.
 */
public void setLearningRate( double alpha )
{
	for (int n=0; n<numNets; n++)
		((MLP)networks.get(n)).learningRate=alpha;
}     

////////////////////////////////////////////////////////////////////////////////////////


/**
 * Sets the status of the <code>voting</code> flag.
 * 
 * @param value - Boolean flag indicating whether to use voting or not.
 */
public void setVoting( boolean value )
{
	voting = value;
}

////////////////////////////////////////////////////////////////////////////////////////

/**
 * Sets the status of the <code>classifier</code> flag.
 * 
 * @param value - Boolean flag indicating whether to be a classifier or not.
 */
public void setClassificationEnsemble( boolean value )
{
	classificationEnsemble=value;
	for (int n=0; n<numNets; n++)
		((MLP)networks.get(n)).sigmoidOutput=value;
}

////////////////////////////////////////////////////////////////////////////////////////

}
