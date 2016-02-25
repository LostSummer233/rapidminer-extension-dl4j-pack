package com.rapidminerchina.extension.dl4j.learners;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;

import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.Tools;
import com.rapidminer.operator.Model;
import com.rapidminer.operator.OperatorCapability;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminerchina.extension.dl4j.layers.AbstractLayer;
import com.rapidminerchina.extension.dl4j.layers.ConvolutionalLayer;
import com.rapidminerchina.extension.dl4j.layers.OutputLayer;
import com.rapidminerchina.extension.dl4j.layers.SubSamplingLayer;
import com.rapidminerchina.extension.dl4j.model.MultiLayerNetModel;

public class SimpleNeuralNetwork extends AbstractDLModelLearner {
	
	public SimpleNeuralNetwork(OperatorDescription description) {
		super(description);
		// TODO Auto-generated constructor stub
	}

	@Override
	public Model learn(ExampleSet exampleSet) throws OperatorException {
		
		Tools.onlyNonMissingValues(exampleSet, getOperatorClassName(), this, new String[0]);
		
		MultiLayerNetModel model = new MultiLayerNetModel(exampleSet);
		
		// retrieve information
		// for the whole model
		
		// iteration
		int iteration = getParameterAsInt(PARAMETER_ITERATION);
		
		// learning rate, decay and momentum
		double learningRate = getParameterAsDouble(PARAMETER_LEARNING_RATE);
		double decay = getParameterAsDouble(PARAMETER_DECAY);
		double momentum = getParameterAsDouble(PARAMETER_MOMENTUM);
		
		// optimize function
		int optimizationAlgorithmIndex = getParameterAsInt(PARAMETER_OPTIMIZATION_ALGORITHM);
		OptimizationAlgorithm optimizationAlgorithm = getOptimizationAlgorithm(optimizationAlgorithmIndex);
		
		// for expert features
		// shuffle
		boolean shuffle = getParameterAsBoolean(PARAMETER_SHUFFLE);
		
		// normalize
		boolean normalize = getParameterAsBoolean(PARAMETER_NORMALIZE);
		
		// regularization
		boolean regularization = getParameterAsBoolean(PARAMETER_REGULARIZATION);
		double l1 = getParameterAsDouble(PARAMETER_L1);
		double l2 = getParameterAsDouble(PARAMETER_L2);
		
		// mimibatch
		boolean miniBatch = getParameterAsBoolean(PARAMETER_MINIBATCH);
		
		// minimize loss function
		boolean minimize = getParameterAsBoolean(PARAMETER_MINIMIZE);
		
		// seed
//		boolean specifySeed = getParameterAsBoolean(PARAMETER_USE_LOCAL_RANDOM_SEED);
		long seed = getParameterAsInt(PARAMETER_LOCAL_RANDOM_SEED);
		
		// set up the configurations
		NeuralNetConfiguration.Builder configBuilder = new NeuralNetConfiguration.Builder()
				.iterations(iteration)
				.learningRate(learningRate)
        		.learningRateScoreBasedDecayRate(decay)
        		.momentum(momentum)
        		.optimizationAlgo(optimizationAlgorithm)
        		.regularization(regularization)
        		.miniBatch(miniBatch)
        		.minimize(minimize)
        		.seed(seed);
		
		if (regularization){
			configBuilder.setL1(l1);
			configBuilder.setL1(l2);			
		}
		
		ListBuilder listBuilder = configBuilder.list(structure.size());
		
		int inSize = exampleSet.getAttributes().size();
		
		for (int i=0; i<structure.size(); i++){
			AbstractLayer layer = structure.get(i);
			
			if (i==structure.size()-1) {
				
				if(layer.getClass() == OutputLayer.class){

					listBuilder.layer(i,((OutputLayer)layer).getLayer(inSize,
							exampleSet.getAttributes().getLabel().getMapping().getValues().size()));
				} else {
					throw new OperatorException("Please put an output layer in the end of the neural network");
				}
				
			} else {
				
				if (layer.getClass() == ConvolutionalLayer.class 
						|| layer.getClass() == SubSamplingLayer.class){
					throw new OperatorException("Convolutional layers and subsampling layers are not "
							+ "supported in the General Neural Network: "
							+ this.getName() + ", please use Convolutional Neural Network, instead.");
				}
				
				listBuilder.layer(i,layer.getLayer(inSize));
				inSize = layer.getNumNodes();
				
			}
		}
		
        // construct the configuration information and train the model
	    MultiLayerConfiguration config = listBuilder.build();
		model.train(exampleSet, config, shuffle, normalize);
		return model;
	}

}
