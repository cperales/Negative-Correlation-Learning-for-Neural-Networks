package dataprocessing;

/**
 * A single input-output pattern pairing.
 */
public class Example
{
	public double [] inputs;
	public double target;

	///////////////////////////////////////////////////////////////
	
	public Example( int numInputs )
	{
		inputs = new double[numInputs];
		target = 0;
	}
	
	///////////////////////////////////////////////////////////////

	public void print()	{ System.out.println(toString()); }
	public String toString()
	{
		String output = "{ ";
		for (int i=0; i<inputs.length; i++)
			output+=inputs[i]+"  ";
		output+="}";
		
		output+=" = "+target;

		return output;
	}

	///////////////////////////////////////////////////////////////

	public int getClassification()
	{
		return (int)Math.round(target);
	}

	///////////////////////////////////////////////////////////////

	public boolean equals( Object other )
	{
		Example e = (Example)other;
		if(e.inputs.equals(this.inputs) && e.target==this.target)
			return true;
		else
			return false;
	}

	///////////////////////////////////////////////////////////////
}
