package com.rapidminerchina.extension.dl4j.learners;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Layer.TrainingMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;

import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.Tools;
import com.rapidminer.gui.tools.syntax.InputHandler.insert_char;
import com.rapidminer.operator.Model;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.tools.LogService;
import com.rapidminerchina.extension.dl4j.layers.AbstractLayer;
import com.rapidminerchina.extension.dl4j.layers.ConvolutionalLayer;
import com.rapidminerchina.extension.dl4j.layers.OutputLayer;
import com.rapidminerchina.extension.dl4j.model.MultiLayerNetModel;

public class ConvolutionalNeuralNetwork extends AbstractDLModelLearner {
	
	/**
	 * The parameter name for &quot;Width of image.&quot;
	 */
	public static final String PARAMETER_WIDTH = "width_of_image"; 
	
	/**
	 * The parameter name for &quot;Height of image.&quot;
	 */
	public static final String PARAMETER_HEIGHT = "height_of_image"; 

	/**
	 * The parameter name for &quot;Depth/number of channels of image.&quot;
	 */
	public static final String PARAMETER_DEPTH = "depth/number_of_channels_of_image"; 
	
	public ConvolutionalNeuralNetwork(OperatorDescription description) {
		super(description);
		// TODO Auto-generated constructor stub
	}

	@Override
	public List<ParameterType> getParameterTypes(){
		
		List<ParameterType> types = super.getParameterTypes();
		
		types.add(1, new ParameterTypeInt(PARAMETER_WIDTH,
						"width of image",
						1,Integer.MAX_VALUE,28));
		
		types.add(2, new ParameterTypeInt(PARAMETER_HEIGHT,
						"height of image",
						1,Integer.MAX_VALUE,28));
		
		types.add(3, new ParameterTypeInt(PARAMETER_DEPTH,
						"depth/number of channels of image",
						1,Integer.MAX_VALUE,3));
		
		return types;
	}
	
	@Override
	public Model learn(ExampleSet exampleSet) throws OperatorException {

		Tools.onlyNonMissingValues(exampleSet, getOperatorClassName(), this, new String[0]);
		
		MultiLayerNetModel model = new MultiLayerNetModel(exampleSet);
		
		// retrieve information
		// for the whole model
		
		// iteration
		int iteration = getParameterAsInt(PARAMETER_ITERATION);
		
		// image size and depth
		int width = getParameterAsInt(PARAMETER_WIDTH);
		int height = getParameterAsInt(PARAMETER_HEIGHT);
		int depth = getParameterAsInt(PARAMETER_DEPTH);
		
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
		
		List<String> layerNames = new ArrayList<String>();
		int inDepth = depth;
		int[] inSize = new int[]{width, height};
		
		/*
		 * check the number of attributes in the input example set is suitable for the convolutional neural net,
		 * i.e. #attributes = width * height * depth
		 */
		if (exampleSet.getAttributes().size() != inSize[0] * inSize[1] * inDepth){
			throw new OperatorException("The input attribute size does not match the parameters defined in "
					+ "the convolutional neural network "
					+ this.getName() 
					+ ". Please ensure that width * height * depth equals to the input attribute size.");
		}
		
		for (int i=0; i<structure.size(); i++){
			
			AbstractLayer layer = structure.get(i);
			
			if (i==structure.size()-1) {
				
				if(layer.getClass() == OutputLayer.class){

					listBuilder.layer(i,((OutputLayer)layer).getLayer(false,
							exampleSet.getAttributes().getLabel().getMapping().getValues().size()));
					layerNames.add(layer.getLayerName());
					
				} else {
					throw new OperatorException("Please ensure an output layer in the end of the "
							+ "convolutional neural network "
							+ this.getName() +".");
				}
				
			} else {
				
				if (layer.getClass() == ConvolutionalLayer.class){
					
					int[] outSize = ((ConvolutionalLayer) layer).getOutSize(inSize);
					
					
					if (outSize.length != 2 || outSize[0] < 1 || outSize[1] < 1){
						
						throw new OperatorException("There is a problem with the parameters of layer " + layer.getName()
						+ ", in the convulutional neural network " + this.getName()
						+ ", please check and make sure that "
						+ "the size of input W, the padding P, the stride S and the "
						+ "size of receptive field F satisfies the fomula "
						+ "2P + W >= n*S + F, with integer n at least 1"
						+ "in both horizental and vertical directions");
					}
					
					listBuilder.layer(i,layer.getLayer(inDepth));
					inDepth = layer.getNumNodes();
					inSize = outSize;
					layerNames.add(layer.getLayerName());
					
				} else {
					
					listBuilder.layer(i, layer.getLayer());
					layerNames.add(layer.getLayerName());
				}
			}
		}
		
		listBuilder.backprop(true).pretrain(false);
		
		new ConvolutionLayerSetup(listBuilder,width,height,depth);
		// construct the configuration information and train the model
		
	    MultiLayerConfiguration config = listBuilder.build();
		model.train(exampleSet, config, shuffle, normalize,layerNames);
		
		return model;
	}

}
