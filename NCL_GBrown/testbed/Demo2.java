package testbed;

import java.util.ArrayList;

import dataprocessing.*;
import predictors.*;


public class Demo2 {


	public static void main(String[] args) {

		System.out.println("Negative Correlation Learning Demo");
		System.out.println("(c) Gavin Brown 2007");
		System.out.println("\nRegression : Friedman's function\n");
		
		DataReader reader = new DataReader("friedman.csv");
		reader.shuffle();
		ArrayList trainingData = reader.getTrainingData();
		ArrayList testingData = reader.getTestingData();

		//////////////////////////////////////////////////////////////
		
		Ensemble ens = new Ensemble();
		for (int i=0; i<5; i++)
			ens.add( new MLP(reader.numInputs, 5, MLP.LINEAR));
		
		System.out.println("Please wait... training simple ensemble (NC lambda = 0)" );
		ens.lambda = 0.0;
		ens.train(trainingData, 1000);
		System.out.println("mse = "+ens.test(testingData)+"\n" );
		
		///////////////////////////////////////////////////////////////
		
		ens = new Ensemble();
		for (int i=0; i<5; i++)
			ens.add( new MLP(reader.numInputs, 5, MLP.LINEAR));
	
		ens.lambda = 0.9;
		System.out.println("Please wait... training with NC, lambda = "+ens.lambda );
		ens.train(trainingData, 1000);
		System.out.println("mse = "+ens.test(testingData)+"\n" );

		////////////////////////////////////////////////////////////////
	}

}
