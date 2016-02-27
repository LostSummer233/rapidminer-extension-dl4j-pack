package com.rapidminerchina.extension.dl4j.layers;

import java.util.List;

import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.conf.layers.RBM.HiddenUnit;
import org.deeplearning4j.nn.conf.layers.RBM.VisibleUnit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeBoolean;
import com.rapidminer.parameter.ParameterTypeCategory;
import com.rapidminer.parameter.ParameterTypeDouble;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.ParameterTypeString;
import com.rapidminer.parameter.UndefinedParameterError;
import com.rapidminer.parameter.conditions.BooleanParameterCondition;


/* *******************************************
 * java doc for title later
 */


/**
 * <ul>
 * <li> the name of the layer;</li>
 * <li> the number of nodes;</li>
 * <li> the type of hidden/output and visible/input nodes;</li>
 * <li> the activation function;</li>
 * <li> the loss function;</li>
 * <li> the dropout rate;</li>
 * <li> the updater;</li>
 * </ul>
 */

public class RBMLayer extends AbstractLayer{

	private String name = "";

	private int numNodes = 0;

	private String activation = null;

	private LossFunction loss = null;

	private VisibleUnit visible = null;

	private HiddenUnit hidden = null;

	private boolean dropout = false;

	private double dropoutrate = 0;

	private Updater updater = null;
	
	/**
	 * The parameter name for &quot;Name of this layer.&quot;
	 */
	public static final String PARAMETER_NAME = "name";
	
	/**
	 * The parameter name for &quot;Number of nodes of this layer.&quot;
	 */
	public static final String PARAMETER_NUMEBR_OF_NODE = "number_of_nodes";
		
	/**
	 * The parameter name for &quot;Type of visible/input nodes of this layer.&quot;
	 */
	public static final String PARAMETER_VISIBLE_TYPE = "type of visible node";
	
	/**
	 * The parameter name for &quot;Type of hidden/output nodes of this layer.&quot;
	 */
	public static final String PARAMETER_HIDDEN_TYPE = "type of hidden node";
	
	/**
	 * The category &quot;Visible node type names&quot;
	 */
	public static final String[] VISIBLE_NODE_TYPE_NAMES = new String[]{
			"linear"
			,"binary"
			,"gaussian"
			,"softmax"
	};
	
	/**
	 * The category &quot;Hidden node type names&quot;
	 */
	public static final String[] HIDDEN_NODE_TYPE_NAMES = new String[]{
			"rectified"
			,"binary"
			,"gaussian"
			,"softmax"
	};
	
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
	
	/** 
	 * Indicates if to use dropout. 
	 */
	public static final String PARAMETER_DROPOUT = "dropout";
	
	/**
	 * The parameter name for &quot;Dropout rate.&quot;
	 */
	public static final String PARAMETER_DROPOUT_RATE = "dropout_rate";
	
	/**
	 * The parameter name for &quot;Updater for each hidden layer.&quot;
	 */
	public static final String PARAMETER_UPDATER = "updater";
	
	/**
	 * The category &quot;Updater&quot;
	 */
	public static final String[] UPDATER_NAMES = new String[]{
			"SGD"
			,"ADAM"
			,"ADADelta"
			,"Nesterovs"
			,"ADAGrad"
			,"RMSProp"
			,"none"
//			,"custom"
	};
	
	
	public RBMLayer(OperatorDescription description) {
		super(description);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public List<ParameterType> getParameterTypes() {
		
		List<ParameterType> types = super.getParameterTypes();
		
		ParameterType type = null;
		
		types.add(new ParameterTypeString(PARAMETER_NAME,
				"The name of this layer",
				"RBM Layer"
				));
		
		types.add(new ParameterTypeInt(PARAMETER_NUMEBR_OF_NODE,
				"The number of nodes in this layer",
				1,Integer.MAX_VALUE,10
				));
		
		types.add(new ParameterTypeCategory(PARAMETER_ACTIVATION_FUNCTION,
				"The activation function of this layer",
				ACTIVATION_FUNCTION_NAMES,
				2));
		
		types.add(new ParameterTypeCategory(PARAMETER_LOSS_FUNCTION,
				"The loss function of this layer",
				LOSS_FUNCTION_NAMES,
				0));
		
		types.add(new ParameterTypeCategory(PARAMETER_VISIBLE_TYPE,
				"The types of visible/input nodes",
				VISIBLE_NODE_TYPE_NAMES,
				2));
		
		types.add(new ParameterTypeCategory(PARAMETER_HIDDEN_TYPE,
				"The type of hidden/output nodes",
				HIDDEN_NODE_TYPE_NAMES,
				2));
		
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

		type = new ParameterTypeCategory(PARAMETER_UPDATER,
				"The updater for this layer",
				UPDATER_NAMES,
				0);
		types.add(type);
		
		return types;
	}
	
	public RBM.Builder generateBuilder() throws UndefinedParameterError{

		name = getParameterAsString(PARAMETER_NAME);
		numNodes = getParameterAsInt(PARAMETER_NUMEBR_OF_NODE);
		
		activation = getParameterAsString(PARAMETER_ACTIVATION_FUNCTION);
		
		int lossIndex = getParameterAsInt(PARAMETER_LOSS_FUNCTION);
		loss = getLossFunction(lossIndex);
		
		int visibleIndex = getParameterAsInt(PARAMETER_VISIBLE_TYPE);
		visible = getVisibleUnit(visibleIndex);
		
		int hiddenIndex = getParameterAsInt(PARAMETER_HIDDEN_TYPE);
		hidden = getHiddenUnit(hiddenIndex);
		
		dropout = getParameterAsBoolean(PARAMETER_DROPOUT);
		dropoutrate = getParameterAsDouble(PARAMETER_DROPOUT_RATE);
		
		int updaterIndex = getParameterAsInt(PARAMETER_UPDATER);
		updater = getUpdater(updaterIndex);
		
		
		RBM.Builder builder = new RBM.Builder(hidden,visible)
				.nOut(numNodes)
				.activation(activation)
				.lossFunction(loss)
				.updater(updater)
				.weightInit(WeightInit.XAVIER) 
				.k(1);
		
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
		RBM.Builder builder = generateBuilder().nIn(i);
		return builder.build();
	}

	@Override
	public int getNumNodes() throws UndefinedParameterError {
		if (numNodes != 0){
			return numNodes;
		} else {
			getLayer();
			return numNodes;
		}
	}

	private LossFunctions.LossFunction getLossFunction(int i){
		switch (i) {
		case 0 :
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
	
	private RBM.HiddenUnit getHiddenUnit(int i){
		switch (i) {
		case 0 : 
			return RBM.HiddenUnit.RECTIFIED;
		case 1 : 
			return RBM.HiddenUnit.BINARY;
		case 2 : 
			return RBM.HiddenUnit.GAUSSIAN;
		case 3 :
			return RBM.HiddenUnit.SOFTMAX;
		default :
			return null;
		}	
	}
	
	private RBM.VisibleUnit getVisibleUnit(int i){
		switch (i) {
		case 0 : 
			return RBM.VisibleUnit.LINEAR;
		case 1 : 
			return RBM.VisibleUnit.BINARY;
		case 2 : 
			return RBM.VisibleUnit.GAUSSIAN;
		case 3 : 
			return RBM.VisibleUnit.SOFTMAX;
		default :
			return null;
		}	
	}
	
	private Updater getUpdater(int i){
		switch (i) {
		case 0:
			return Updater.SGD;
		case 1:
			return Updater.ADAM;
		case 2:
			return Updater.ADADELTA;
		case 3:
			return Updater.NESTEROVS;
		case 4:
			return Updater.ADAGRAD;
		case 5:    
			return Updater.RMSPROP;
		case 6:
			return Updater.NONE;
		default:
			return null;
		}
	}
	
	@Override
	public String getLayerName() {
		return name;
	}
}
