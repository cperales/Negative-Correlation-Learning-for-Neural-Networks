package dataprocessing;


import java.io.*;
import java.util.*;

/**
 * General purpose datafile interface.  Handles all data manipulations, e.g. cross-validation.
 */
public class DataReader
{
	public ArrayList<Example> exampleList = null;

	public int numInputs    = -1;
	public double trainingFraction = 0.5; // half training, half testing

//////////////////////////////////////////////////////////////////////////////////////

	public DataReader ( String filename )
	{
		BufferedReader myReader = null;
		
		try
		{
			// Attempt to open the file
			//
			myReader = new BufferedReader( new FileReader( new File( filename ) ) );
		}
		catch( FileNotFoundException ex )
		{
			System.err.println( "Datafile '"+filename+"' not found." );
			System.out.print(ex);
			System.exit(1);
		}
		
		
		//Initialise the data structure to hold our training or testing examples
		//
		exampleList = new ArrayList<Example>();
		

		try
		{
			// Loop round this while we have not reached the end of the file
			//
			while (myReader.ready())
			{	
				//Read one line from the file (corresponding to one Example)
				//
				String line = myReader.readLine();
			
				//Break that line up into chunks separated by commas, spaces or tabs
				//
				StringTokenizer myTokenizer = new StringTokenizer( line, ", \t" );
				
				//Initialise a data structure to hold this particular Example
				//and take away 1 since the last item is the target
				//
				if(numInputs==-1) numInputs = myTokenizer.countTokens()-1;
				
				Example thisExample = new Example( numInputs );

				//Loop through each chunk of the line we read from the file, adding to our data structure
				//
				int attrib=0;
				while (attrib < numInputs)
				{
					String token = myTokenizer.nextToken();
					
					thisExample.inputs[attrib++] = Double.parseDouble( token );
				}

				//Now read the target value
				//
				thisExample.target = Double.parseDouble( myTokenizer.nextToken() );

				//Add this Example to our list of examples
				//
				exampleList.add( thisExample );
			}
		}
		catch (IOException ioe)
		{
			System.err.println( "IO Exception when reading datafile '"+filename+"'." );
			System.exit(1);
		}
	
	}
	
//////////////////////////////////////////////////////////////////////////////////////
	
	public int numExamples()
	{
		return exampleList.size();
	}
	
//////////////////////////////////////////////////////////////////////////////////////

	public ArrayList getData()
	{
		return exampleList;
	}
	
//////////////////////////////////////////////////////////////////////////////////////

	public void normalize()
	{
		ArrayList data = exampleList;
		
		Example means = new Example(numInputs);
		Example stdev = new Example(numInputs);

		//FIND THE MEANS
		//
		for (int whichExample=0; whichExample<data.size(); whichExample++)
		{
			Example ex = (Example)data.get(whichExample);
			for (int i=0; i<means.inputs.length; i++)
				means.inputs[i] += ex.inputs[i];	
		}
		for (int i=0; i<means.inputs.length; i++)
			means.inputs[i] /= data.size();
		
		
		//FIND THE STDEVS
		//
		for (int whichExample=0; whichExample<data.size(); whichExample++)
		{
			Example ex = (Example)data.get(whichExample);
			for (int i=0; i<stdev.inputs.length; i++)
				stdev.inputs[i] += (ex.inputs[i]-means.inputs[i])*(ex.inputs[i]-means.inputs[i]);
		}
		for (int i=0; i<stdev.inputs.length; i++)
			stdev.inputs[i] = Math.sqrt( stdev.inputs[i] / (data.size()-1) );		
		
		
		//NORMALIZE
		//
		for (int whichExample=0; whichExample<data.size(); whichExample++)
		{
			Example ex = (Example)data.get(whichExample);
			for (int i=0; i<ex.inputs.length; i++)
				ex.inputs[i] = (ex.inputs[i]-means.inputs[i]) / stdev.inputs[i];	
		}
	}
	
//////////////////////////////////////////////////////////////////////////////////////
	
	public void shuffle()
	{
		ArrayList<Example> shuffled = new ArrayList<Example>();
		Random gen = new Random();
		
		//System.out.println("before: "+exampleList.size());
		
		while (exampleList.size() > 0)
		{
			int eg = gen.nextInt(exampleList.size());
			shuffled.add( exampleList.get(eg) );
			exampleList.remove(eg);
		}
		
		//System.out.println("after: "+shuffled.size());
		
		exampleList = shuffled;
	}
	
//////////////////////////////////////////////////////////////////////////////////////
	
	public ArrayList getTrainingData()
	{
		int numTraining = (int)(exampleList.size() * trainingFraction);
		
		ArrayList<Example> training = new ArrayList<Example>();
		
		for (int i=0; i<exampleList.size(); i++)
			training.add( exampleList.get(i) );

		return training;
	}

//////////////////////////////////////////////////////////////////////////////////////
	
	public ArrayList getTestingData()
	{
		int numTraining = (int)(exampleList.size() * trainingFraction);
		
		ArrayList<Example> testing = new ArrayList<Example>();
		
		for (int i=numTraining; i<exampleList.size(); i++)
			testing.add( exampleList.get(i) );
			
		return testing;
	}

//////////////////////////////////////////////////////////////////////////////////////
	
	public void printData()
	{
		for (int whichExample=0; whichExample<numExamples(); whichExample++)
		{
			//Retrieve the Example at index number 'whichExample'
			//
			Example thisExample = (Example)exampleList.get(whichExample);
			
			//Print it
			//
			thisExample.print();
		}
	}

//////////////////////////////////////////////////////////////////////////////////////	

	public void addTargetNoise(double prob)
	{
		for (int whichExample=0; whichExample<exampleList.size(); whichExample++)
		{
			//Retrieve the Example at index number 'whichExample'
			//
			Example ex = (Example)exampleList.get(whichExample);
			
			if(Math.random() < prob)
			{
				//corrupt the data
				//
				if(ex.target==1) ex.target=0; else ex.target=1;
			}
		}
	}


//////////////////////////////////////////////////////////////////////////////////////	
	
	
}

