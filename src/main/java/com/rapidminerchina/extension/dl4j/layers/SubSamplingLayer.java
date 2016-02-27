package com.rapidminerchina.extension.dl4j.layers;

import java.util.List;

import org.deeplearning4j.nn.conf.layers.Layer;

import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeCategory;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.ParameterTypeString;
import com.rapidminer.parameter.UndefinedParameterError;

public class SubSamplingLayer extends AbstractLayer {

	private String name = "";

	private org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType poolingType = null;

	private int[] size = new int[]{0,0};
	
	/**
	 * The parameter name for &quot;Name of this layer.&quot;
	 */
	public static final String PARAMETER_NAME = "name";
	
	/**
	 * The parameter name for &quot;Pooling type.&quot;
	 */
	public static final String PARAMETER_POOLING_TYPE = "pooling_type";
	
	/**
	 * The category &quot;Pooling types&quot;
	 */
	public static final String[] POOLING_TYPE_NAMES = new String[]{
		"max"
		,"average"
		,"sum"
	};
	
	/**
	 * The width of the filter
	 */
	public static final String PARAMETER_WIDTH = "width_of_filter";
	
	/**
	 * The height of the filter
	 */
	public static final String PARAMETER_HEIGHT = "height_of_filter";	
	
	public SubSamplingLayer(OperatorDescription description) {
		super(description);
		// TODO Auto-generated constructor stub
	}
	
	public List<ParameterType> getParameterTypes(){

		List<ParameterType> types = super.getParameterTypes();
		
		types.add(new ParameterTypeString(PARAMETER_NAME,
				"the name of this layer",
				"Subsampleing Layer"
				));
		
		types.add(new ParameterTypeCategory(PARAMETER_POOLING_TYPE,
				"the type of pooling, often, we use max as the pooling method",
				POOLING_TYPE_NAMES,
				0));
		
		types.add(new ParameterTypeInt(PARAMETER_WIDTH,
				"the width of the filter",
				1,Integer.MAX_VALUE,2));
		
		types.add(new ParameterTypeInt(PARAMETER_HEIGHT,
				"the height of the filter",
				1,Integer.MAX_VALUE,2));
		
		return types;
	}

	public org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder generateBuilder() throws UndefinedParameterError{
		
		name = getParameterAsString(PARAMETER_NAME);
		
		int poolingTypeIndex = getParameterAsInt(PARAMETER_POOLING_TYPE);
		poolingType = getpoolingType(poolingTypeIndex);
		
		size[0] = getParameterAsInt(PARAMETER_WIDTH);
		size[1] = getParameterAsInt(PARAMETER_HEIGHT);
		
		org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder builder =
				new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(
						poolingType, size);
		
		return builder;
	}
	
	/**
	 * This will be the mostly used method
	 */
	@Override
	public Layer getLayer() throws UndefinedParameterError {
		return generateBuilder().build();
	}

	@Override
	public Layer getLayer(int i) throws UndefinedParameterError {
		// simple ignore the input i because it is meaningless.
		return getLayer();
	}

	/**
	 * Never used
	 */
	@Override
	public int getNumNodes() throws UndefinedParameterError {
		return 0;
	}

	private org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType getpoolingType(int i){
		switch (i) {
		case 0:
			return org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType.MAX;
		case 1:
			return org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType.AVG;
		case 2:
			return org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType.SUM;
		default:
			return null;
		}
	}
	
	@Override
	public String getLayerName() {
		return name;
	}
}
