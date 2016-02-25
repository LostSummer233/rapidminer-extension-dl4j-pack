package com.rapidminerchina.extension.dl4j.layers;

import java.util.List;
import java.util.logging.Level;

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;

import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeBoolean;
import com.rapidminer.parameter.ParameterTypeCategory;
import com.rapidminer.parameter.ParameterTypeDouble;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.ParameterTypeString;
import com.rapidminer.parameter.UndefinedParameterError;
import com.rapidminer.parameter.conditions.BooleanParameterCondition;
import com.rapidminer.tools.LogService;
import com.rapidminer.tools.expression.internal.function.rounding.Floor;

public class ConvolutionalLayer extends AbstractLayer {

	private String name = "";
	
	private int depth = 0;

	private boolean differentSize = false;
	
	private int[] size = new int[]{0,0};
	
	private boolean differentStride = false;
	
	private int[] stride = new int[]{1,1};
	
	private boolean differentPadding = false;
	
	private int[] padding = new int[]{0,0};

	private String activation = null;

	private boolean dropout = false;

	private double dropoutrate = 0;

	/**
	 * The parameter name for &quot;Name of this layer.&quot;
	 */
	public static final String PARAMETER_NAME = "name";
	
	/**
	 * The parameter name for &quot;The depth of this layer.&quot;
	 */
	public static final String PARAMETER_DEPTH = "depth";
	
	/**
	 * Indicate if the receptive field is square or the width and height is specified separately.
	 */
	public static final String PARAMETER_DIFFERENT_SIZE = "specify_width_and_height_separately_for_receptive_field";
	
	/**
	 * The parameter name for &quot;The size of receptive field.&quot;
	 */
	public static final String PARAMETER_SIZE = "the_size_of_receptive_field";
	
	/**
	 * The parameter name for &quot;The width of receptive field.&quot;
	 */
	public static final String PARAMETER_WIDTH = "the_width_of_receptive_field";
	
	/**
	 * The parameter name for &quot;The height of receptive field.&quot;
	 */
	public static final String PARAMETER_HEIGHT = "the_height_of_receptive_field";
	
	/**
	 * Indicate if the stride is the same for both width and height.
	 */
	public static final String PARAMETER_DIFFERENT_STRIDE = "specify_different_stride_in_width_and_height";
	
	/**
	 * The parameter name for &quot;The stride.&quot;
	 */
	public static final String PARAMETER_STRIDE = "the_stride";
	
	/**
	 * The parameter name for &quot;The stride in width.&quot;
	 */
	public static final String PARAMETER_STRIDE_IN_WIDTH = "the_stride_in_width";
	
	/**
	 * The parameter name for &quot;The stride in height.&quot;
	 */
	public static final String PARAMETER_STRIDE_IN_HEIGHT = "the_stride_in_height";
	
	/**
	 * Indicate if the padding is the same for both width and height
	 */
	public static final String PARAMETER_DIFFERENT_PADDING = "specify_different_padding_in_width_and_height";
	
	/**
	 * The parameter name for &quot;The length of padding.&quot;
	 */
	public static final String PARAMETER_PADDING = "the_length_of_padding";
	
	/**
	 * The parameter name for &quot;The padding in width.&quot;
	 */
	public static final String PARAMETER_PADDING_IN_WIDTH = "the_padding_in_width";
	
	/**
	 * The parameter name for &quot;The padding in height.&quot;
	 */
	public static final String PARAMETER_PADDING_IN_HEIGHT = "the_padding_in_height";
	
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
	 * Indicates if to use dropout. 
	 */
	public static final String PARAMETER_DROPOUT = "dropout";
	
	/**
	 * The parameter name for &quot;Dropout rate.&quot;
	 */
	public static final String PARAMETER_DROPOUT_RATE = "dropout_rate";
	
	public ConvolutionalLayer(OperatorDescription description) {
		super(description);
		// TODO Auto-generated constructor stub
	}

	@Override
	public List<ParameterType> getParameterTypes() {
		
		List<ParameterType> types = super.getParameterTypes();
		
		ParameterType type = null;
		
		types.add(new ParameterTypeString(PARAMETER_NAME,
				"The name of this layer",
				"Convolutional Layer"
				));
		
		types.add(new ParameterTypeInt(PARAMETER_DEPTH,
				"The depth of this layer",
				1,Integer.MAX_VALUE,6
				));
		
		types.add(new ParameterTypeBoolean(PARAMETER_DIFFERENT_SIZE,
				"Use different width and height for the receptive field",
				false));
		
		type = new ParameterTypeInt(PARAMETER_SIZE,
				"The side length of the receptive field",
				1,Integer.MAX_VALUE,5);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_SIZE, 
						false, false));
		types.add(type);
		
		type = new ParameterTypeInt(PARAMETER_WIDTH,
				"The width of the receptive field",
				1,Integer.MAX_VALUE,5);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_SIZE, 
						false, true));
		types.add(type);
		
		type = new ParameterTypeInt(PARAMETER_HEIGHT,
				"The height of the receptive field",
				1,Integer.MAX_VALUE,5);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_SIZE, 
						false, true));
		types.add(type);
		
		types.add(new ParameterTypeBoolean(PARAMETER_DIFFERENT_STRIDE,
				"Use different stride in width and height",
				false));
		
		type = new ParameterTypeInt(PARAMETER_STRIDE,
				"The stride length",
				1,Integer.MAX_VALUE,3);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_STRIDE, 
						false, false));
		types.add(type);
		
		type = new ParameterTypeInt(PARAMETER_STRIDE_IN_WIDTH,
				"The stride in width",
				1,Integer.MAX_VALUE,5);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_STRIDE, 
						false, true));
		types.add(type);
		
		type = new ParameterTypeInt(PARAMETER_STRIDE_IN_HEIGHT,
				"The stride in height",
				1,Integer.MAX_VALUE,5);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_STRIDE, 
						false, true));
		types.add(type);
		
		types.add(new ParameterTypeBoolean(PARAMETER_DIFFERENT_PADDING,
				"Use different padding in width and height",
				false));
		
		type = new ParameterTypeInt(PARAMETER_PADDING,
				"The padding size",
				0,Integer.MAX_VALUE,3);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_PADDING, 
						false, false));
		types.add(type);
		
		type = new ParameterTypeInt(PARAMETER_PADDING_IN_WIDTH,
				"The padding in width",
				0,Integer.MAX_VALUE,5);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_PADDING, 
						false, true));
		types.add(type);
		
		type = new ParameterTypeInt(PARAMETER_PADDING_IN_HEIGHT,
				"The padding in height",
				0,Integer.MAX_VALUE,5);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DIFFERENT_PADDING, 
						false, true));
		types.add(type);
		
		types.add(new ParameterTypeCategory(PARAMETER_ACTIVATION_FUNCTION,
				"The activation function of this layer",
				ACTIVATION_FUNCTION_NAMES,
				0));
		
		type = new ParameterTypeBoolean(PARAMETER_DROPOUT,
				"Indicates if to use dropout, using dropout helps overcome overfitting, but may disturb converge for very small network",
				false);
		type.setExpert(true);
		types.add(type);
		
		type = new ParameterTypeDouble(PARAMETER_DROPOUT_RATE,
				"The dropout rate for this layers.",
				0d,0.7d,0.5d);
		type.setExpert(true);
		type.registerDependencyCondition(
				new BooleanParameterCondition(this, 
						PARAMETER_DROPOUT, 
						false, true));
		types.add(type);
		
		return types;
	}
	

	public org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder generateBuilder() throws UndefinedParameterError{

		name = getParameterAsString(PARAMETER_NAME);
		depth = getParameterAsInt(PARAMETER_DEPTH);
		
		differentSize = getParameterAsBoolean(PARAMETER_DIFFERENT_SIZE);
		if (differentSize) {
			size[0] = getParameterAsInt(PARAMETER_WIDTH);
			size[1] = getParameterAsInt(PARAMETER_HEIGHT);
		} else {
			size[0] = getParameterAsInt(PARAMETER_SIZE);
			size[1] = getParameterAsInt(PARAMETER_SIZE);
		}
		
		differentStride = getParameterAsBoolean(PARAMETER_DIFFERENT_STRIDE);
		if (differentStride) {
			stride[0] = getParameterAsInt(PARAMETER_STRIDE_IN_WIDTH);
			stride[1] = getParameterAsInt(PARAMETER_STRIDE_IN_HEIGHT);
		} else {
			stride[0] = getParameterAsInt(PARAMETER_STRIDE);
			stride[1] = getParameterAsInt(PARAMETER_STRIDE);
		}
		
		if(differentPadding) {
			padding[0] = getParameterAsInt(PARAMETER_PADDING_IN_WIDTH);
			padding[1] = getParameterAsInt(PARAMETER_PADDING_IN_HEIGHT);
		} else {
			padding[0] = getParameterAsInt(PARAMETER_PADDING);
			padding[1] = getParameterAsInt(PARAMETER_PADDING);
		}
		
		activation = getParameterAsString(PARAMETER_ACTIVATION_FUNCTION);
		
		dropout = getParameterAsBoolean(PARAMETER_DROPOUT);
		dropoutrate = getParameterAsDouble(PARAMETER_DROPOUT_RATE);
		
		org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder builder = 
				new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(size[0],size[1])
				.nOut(depth)
				.stride(stride[0],stride[1])
				.activation(activation)
				.weightInit(WeightInit.XAVIER)
				.padding(padding);
		
		if (dropout){
			builder = builder.dropOut(dropoutrate);
		}
		
		return builder;
	}
	
	@Override
	public Layer getLayer() throws UndefinedParameterError {
		return generateBuilder().build();
	}

	/**
	 * This will be the most used method that reports the configuration of this layer to the NN model it nested in.
	 */
	@Override
	public Layer getLayer(int i) throws UndefinedParameterError {
		org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder builder = generateBuilder().nIn(i);
		return builder.build();
	}

	@Override
	public int getNumNodes() throws UndefinedParameterError {
		if (depth != 0){
			return depth;
		} else {
			getLayer();
			return depth;
		}
	}
	
	public int[] getOutSize(int... inSize) throws UndefinedParameterError{
		
		generateBuilder();
		if (inSize.length != 2) {
			return new int[0];
		} else {
			int[] n = new int [2];
			for (int i=0; i<2; i++){
				n[i] = (int) Math.floor((inSize[i] + 2 * padding[i] - size[i])/stride[i]);
			}
			return n;
		}
		
	}
}
