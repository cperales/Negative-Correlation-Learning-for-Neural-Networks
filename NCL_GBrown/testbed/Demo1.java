package testbed;

import java.util.ArrayList;

import dataprocessing.*;
import predictors.*;


public class Demo1 {


	public static void main(String[] args) {

		System.out.println("Negative Correlation Learning Demo");
		System.out.println("(c) Gavin Brown 2007");
		System.out.println("\nClassification : Heart disease data\n");
		
		DataReader reader = new DataReader("heart.csv");
		reader.shuffle();

		ArrayList trainingData = reader.getTrainingData();
		ArrayList testingData = reader.getTestingData();

		//////////////////////////////////////////////////////////////
		
		Ensemble ens = new Ensemble();
		for (int i=0; i<10; i++)
			ens.add( new MLP(reader.numInputs, 3, MLP.SIGMOID));
		
		ens.setClassificationEnsemble(true);
		
		System.out.println("Please wait... training simple ensemble (NC lambda = 0)" );
		ens.lambda = 0.0;
		ens.train(trainingData, 500);
		System.out.println("error rate = "+ens.test(testingData)+"\n" );
		
		///////////////////////////////////////////////////////////////
		
		ens = new Ensemble();
		for (int i=0; i<10; i++)
			ens.add( new MLP(reader.numInputs, 3, MLP.SIGMOID));
	
		ens.setClassificationEnsemble(true);
		
		System.out.println("Please wait... training with NC, lambda = 1.0" );
		ens.lambda = 1;
		ens.train(trainingData, 500);
		System.out.println("error rate = "+ens.test(testingData)+"\n" );

		////////////////////////////////////////////////////////////////
	}

}
