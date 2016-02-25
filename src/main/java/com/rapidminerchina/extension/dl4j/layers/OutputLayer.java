package com.rapidminerchina.extension.dl4j.layers;

import java.util.List;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeBoolean;
import com.rapidminer.parameter.ParameterTypeCategory;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.ParameterTypeString;
import com.rapidminer.parameter.UndefinedParameterError;
import com.rapidminer.parameter.conditions.BooleanParameterCondition;

public class OutputLayer extends AbstractLayer {

	private String name = "";

	private boolean specifyNumNodes = false;
	
	private int numNodes = 0;

	private String activation = null;

	private LossFunction loss = null;

	/**
	 * The parameter name for &quot;Name of this layer.&quot;
	 */
	public static final String PARAMETER_NAME = "name";
	
	/**
	 * Indicate whether we specify the number of nodes in the output layer manually or compute it via the operator one.
	 */
	public static final String PARAMETER_SPECIFY_NUM_OUTPUT = "whether_to_specify_the_number_of_nodes_manully";
	
	/**
	 * The parameter name for &quot;Number of nodes of this layer.&quot;
	 */
	public static final String PARAMETER_NUMEBR_OF_NODE = "number_of_nodes";
	
	/**
	 * The parameter name for &quot;Activation function for this layer.&quot;
	 */
	public static final String PARAMETER_ACTIVATION_FUNCTION = "activation_function";
	
	/**
	 * The category &quot;Activation function&quot;
	 */
	public static final String[] ACTIVATION_FUNCTION_NAMES = new String[]{
			"relu" 
			,"tanh"
			,"sigmoid"
			,"softmax"
			,"hardtanh"
			,"leakyrelu"
			,"maxout"
			,"softsign"
			,"softplus"
//	        ,"linear"
	};
	
	/**
	 * The parameter name for &quot;Loss function for this layer.&quot;
	 */
	public static final String PARAMETER_LOSS_FUNCTION = "loss_function";
	
	/**
	 * The category &quot;Loss function&quot;
	 */
	public static final String[] LOSS_FUNCTION_NAMES = new String[]{
			"mean squared error"
			,"exponential log likelihood"
			,"cross Entropy"
			,"multiclass cross entropy"
			,"RMSE cross entropy"
			,"squared Loss"
			,"reconstruction cross entropy"
			,"negative log likelihood"
//			,"custom"
	};
	
	public OutputLayer(OperatorDescription description) {
		super(description);
		// TODO Auto-generated constructor stub
	}

	@Override
	public List<ParameterType> getParameterTypes() {
		
		List<ParameterType> types = super.getParameterTypes();
		
		ParameterType type = null;
		
		types.add(new ParameterTypeString(PARAMETER_NAME,
				"The name of this layer",
				"Output Layer"
				));
		
		types.add(new ParameterTypeBoolean(PARAMETER_SPECIFY_NUM_OUTPUT,
				"Whether to specify the number of nodes",
				false
				));
		
		type = new ParameterTypeInt(PARAMETER_NUMEBR_OF_NODE,
				"The number of nodes in this layer",
				1,Integer.MAX_VALUE,10
				);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_SPECIFY_NUM_OUTPUT, 
						false, true));
		types.add(type);
		
		types.add(new ParameterTypeCategory(PARAMETER_ACTIVATION_FUNCTION,
				"The activation function of this layer",
				ACTIVATION_FUNCTION_NAMES,
				3));
		
		types.add(new ParameterTypeCategory(PARAMETER_LOSS_FUNCTION,
				"The loss function of this layer",
				LOSS_FUNCTION_NAMES,
				0));
		
		return types;
	}
	
	public org.deeplearning4j.nn.conf.layers.OutputLayer.Builder generateBuilder() 
			throws UndefinedParameterError{

		name = getParameterAsString(PARAMETER_NAME);
		
		specifyNumNodes = getParameterAsBoolean(PARAMETER_SPECIFY_NUM_OUTPUT);
		numNodes = getParameterAsInt(PARAMETER_NUMEBR_OF_NODE);
		
		activation = getParameterAsString(PARAMETER_ACTIVATION_FUNCTION);
		
		int lossIndex = getParameterAsInt(PARAMETER_LOSS_FUNCTION);
		loss = getLossFunction(lossIndex);
		
		org.deeplearning4j.nn.conf.layers.OutputLayer.Builder builder = 
				new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(loss)
				.nOut(numNodes)
				.activation(activation)
				.lossFunction(loss)
				.weightInit(WeightInit.XAVIER);
		
		return builder;
	}
	
	@Override
	public Layer getLayer() throws UndefinedParameterError {
		return generateBuilder().build();
	}
	
	@Override
	public Layer getLayer(int i) throws UndefinedParameterError {
		org.deeplearning4j.nn.conf.layers.OutputLayer.Builder builder = 
				generateBuilder().nIn(i);
		return builder.build();
	}
	
	/**
	 * Mostly used in CNN with isNumIn = false
	 */
	
	/**
	 * 
	 * @param isNumIn
	 * @param i
	 * @return
	 * @throws UndefinedParameterError
	 */
	public Layer getLayer(boolean isNumIn, int i) throws UndefinedParameterError{
		
		if (isNumIn){
			org.deeplearning4j.nn.conf.layers.OutputLayer.Builder builder = 
					generateBuilder().nIn(i);
			return builder.build();
		} else {
			org.deeplearning4j.nn.conf.layers.OutputLayer.Builder builder = 
					generateBuilder().nOut(i);
			return builder.build();
		}
	}
	
	/**
	 * This method will be mostly used
	 */
	public Layer getLayer(int in, int out) throws UndefinedParameterError {
		org.deeplearning4j.nn.conf.layers.OutputLayer.Builder builder = 
				generateBuilder()
				.nIn(in)
				.nOut(out);
		return builder.build();
	}
	
//	@Override
//	public int getNumNodes() throws UndefinedParameterError {
//		// TODO Auto-generated method stub
//		return 0;
//	}

	private LossFunctions.LossFunction getLossFunction(int i){
		switch (i) {
		case 0:
			return LossFunctions.LossFunction.MSE;
		case 1:
			return LossFunctions.LossFunction.EXPLL;
		case 2:
			return LossFunctions.LossFunction.XENT;
		case 3:
			return LossFunctions.LossFunction.MCXENT;
		case 4:
			return LossFunctions.LossFunction.RMSE_XENT;
		case 5:
			return LossFunctions.LossFunction.SQUARED_LOSS;
		case 6:
			return LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY;
		case 7:
			return LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD;
		default:
			return null;
		}
	}

	/**
	 * Never used!
	 */
	@Override
	public int getNumNodes() throws UndefinedParameterError {
		return 0;
	}
	
}
