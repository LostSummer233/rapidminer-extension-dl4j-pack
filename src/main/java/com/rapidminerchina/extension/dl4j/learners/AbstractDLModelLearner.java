package com.rapidminerchina.extension.dl4j.learners;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.Level;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;

import com.rapidminer.example.ExampleSet;
import com.rapidminer.operator.Model;
import com.rapidminer.operator.Operator;
import com.rapidminer.operator.OperatorCapability;
import com.rapidminer.operator.OperatorChain;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.SimpleOperatorChain;
import com.rapidminer.operator.UserError;
import com.rapidminer.operator.learner.CapabilityCheck;
import com.rapidminer.operator.learner.CapabilityProvider;
import com.rapidminer.operator.meta.PartialExampleSetLearner;
import com.rapidminer.operator.ports.InputPort;
import com.rapidminer.operator.ports.OutputPort;
import com.rapidminer.operator.ports.metadata.SubprocessTransformRule;
import com.rapidminer.parameter.ParameterType;
import com.rapidminer.parameter.ParameterTypeBoolean;
import com.rapidminer.parameter.ParameterTypeCategory;
import com.rapidminer.parameter.ParameterTypeDouble;
import com.rapidminer.parameter.ParameterTypeInt;
import com.rapidminer.parameter.conditions.BooleanParameterCondition;
import com.rapidminer.tools.LogService;
import com.rapidminer.tools.ParameterService;
import com.rapidminerchina.extension.dl4j.io.LayerSemaphore;
import com.rapidminerchina.extension.dl4j.layers.AbstractLayer;
import com.rapidminerchina.extension.dl4j.layers.OutputLayer;

public abstract class AbstractDLModelLearner extends OperatorChain implements CapabilityProvider{
	
	protected InputPort trainPort = getInputPorts().createPort("training examples", ExampleSet.class);
	protected OutputPort modelPort = getOutputPorts().createPort("model");
	protected OutputPort examplePort = getOutputPorts().createPort("examples");

	protected final OutputPort start = getSubprocess(0).getInnerSources().createPort("start");
	protected final InputPort end = getSubprocess(0).getInnerSinks().createPort("end");
	
	protected List<AbstractLayer> structure = new LinkedList<AbstractLayer>();
	

	/**
	 * The parameter name for &quot;The number of training iterations used for the training.&quot;
	 */
	public static final String PARAMETER_ITERATION = "iteration";
	
	/**
	 * The parameter name for &quot;The learning rate determines by how much we change the weights
	 * at each step.&quot;
	 */
	public static final String PARAMETER_LEARNING_RATE = "learning_rate";

	/** 
	 * The parameter name for &quot;The rate that learning rate decreases.&quot;
	 */
	public static final String PARAMETER_DECAY = "decay";
	
	/**
	 * The parameter name for &quot;The momentum simply adds a fraction of the previous weight
	 * update to the current one (prevent local maximum and smoothes optimization directions).&quot;
	 */
	public static final String PARAMETER_MOMENTUM = "momentum";

	/**
	 * The parameter name for &quot;The optimization algorithm of the neural net.&quot;
	 */
	public static final String PARAMETER_OPTIMIZATION_ALGORITHM = "optimiazation_algorithm";

	/**
	 * The category &quot;optimize algorithm&quot;
	 */
	public static final String[] OPTIMIZE_ALGORITHM_NAMES = new String[]{
			"line_gradient_descent"
			,"conjugate_gradient"
			,"lbfgs"
			,"stochastic_gradient_descent"
//			,"hessian_free"
	};
	
	/** 
	 * Indicates if the input data should be shuffled before learning. 
	 */
	public static final String PARAMETER_SHUFFLE = "shuffle";

	/** 
	 * Indicates if the input data should be normalized between -1 and 1 before learning. 
	 */
	public static final String PARAMETER_NORMALIZE = "normalize";
	
	/**
	 * Indicate if to use regularization
	 */
	public static final String PARAMETER_REGULARIZATION = "regularization";
	
	/**
	 * The name for &quot;The weight of l1 regularization.&quot;
	 */
	public static final String PARAMETER_L1 = "l1";
	
	/**
	 * The name for &quot;The weight of l2 regularization.&quot;
	 */
	public static final String PARAMETER_L2 = "l2";
	
	/**
	 * Indicates if to use mini batch.
	 */
	public static final String PARAMETER_MINIBATCH = "mini_batch";
	
	/**
	 * Indicates if to minimize the loss function or maximize.
	 */
	public static final String PARAMETER_MINIMIZE = "minimize_loss_function";
	
	/**
	 * Indicates if to use local random seed.
	 */
	public static final String PARAMETER_USE_LOCAL_RANDOM_SEED = "use_local_random_seed";

	/**
	 * The name for &quot;The value of local random seed.&quot;
	 */
	public static final String PARAMETER_LOCAL_RANDOM_SEED = "local_random_seed";
	
	public AbstractDLModelLearner(OperatorDescription description){
		super(description, "Layer Structure");
		getTransformer().addRule(new SubprocessTransformRule(getSubprocess(0)));
		getTransformer().addGenerationRule(examplePort, ExampleSet.class);
		getTransformer().addGenerationRule(modelPort, Model.class);
		getTransformer().addGenerationRule(start, LayerSemaphore.class);
	}
	
	@Override
	public boolean supportsCapability(OperatorCapability capability) {
		switch (capability) {
			case NUMERICAL_ATTRIBUTES:
			case POLYNOMINAL_LABEL:
			case BINOMINAL_LABEL:
			case NUMERICAL_LABEL:
			case WEIGHTED_EXAMPLES:
				return true;
				// $CASES-OMITTED$
			default:
				return false;
		}
	}
	
	public boolean onlyWarnForNonSufficientCapabilities() {
		return false;
	}
	
	
	/**
	 * Parameters allow for customization
	 */
	@Override
	public List<ParameterType> getParameterTypes() {
		List<ParameterType> types = super.getParameterTypes();
		
		ParameterType type = null;
		
		// general setting
		types.add(new ParameterTypeInt(
				PARAMETER_ITERATION,
				"The number of iterations used for the neural network training.", 
				1, Integer.MAX_VALUE, 500,
				false));

		types.add(new ParameterTypeDouble(
				PARAMETER_LEARNING_RATE,
				"The learning rate determines by how much we change the weights at each step. May not be 0.",
				Double.MIN_VALUE, 1.0d, 0.9,
				false));
		
		types.add(new ParameterTypeDouble(
				PARAMETER_DECAY,
				"The rate that learning rate decreases",
				0.0d, 1.0d, 0.99d,
				false));
		
		types.add(new ParameterTypeDouble(
				PARAMETER_MOMENTUM,
				"The momentum simply adds a fraction of the previous weight update to the current one (prevent local maxima and smoothes optimization directions).",
				0.0d, 1.0d, 0.2d,
				false));

		types.add(new ParameterTypeCategory(
				PARAMETER_OPTIMIZATION_ALGORITHM,
				"The opimization function",
				OPTIMIZE_ALGORITHM_NAMES,
				1,
				false));
	
		// for expert features
		type = new ParameterTypeBoolean(
				PARAMETER_SHUFFLE,
				"Indicates if the input data should be shuffled before learning.",
				true);
		type.setExpert(true);
		types.add(type);

		type = new ParameterTypeBoolean(
				PARAMETER_NORMALIZE,
				"Indicates if the input data should be normalized.",
				true);
		type.setExpert(true);
		types.add(type);
		
		type = new ParameterTypeBoolean(
				PARAMETER_REGULARIZATION,
				"Indicates if to use regularization. This prevent overfitting and balance weights between features",
				false);
		type.setExpert(true);
		types.add(type);
		
		type = new ParameterTypeDouble(
				PARAMETER_L1,
				"The weight on l1 regularization.",
				0d, 1d, 0d);
		type.setExpert(true);
		type.registerDependencyCondition(
				new BooleanParameterCondition(
						this,
						PARAMETER_REGULARIZATION,
						false,true));
		types.add(type);
		
		type = new ParameterTypeDouble(
				PARAMETER_L2,
				"The weight on l2 regularization.",
				0d, 1d, 0d);
		type.setExpert(true);
		type.registerDependencyCondition(
				new BooleanParameterCondition(
						this,
						PARAMETER_REGULARIZATION,
						false,true));
		types.add(type);
		
		type = new ParameterTypeBoolean(
				PARAMETER_MINIBATCH,
				"Indicates if to use miniBatch.",
				true);
		type.setExpert(true);
		types.add(type);
		
		type = new ParameterTypeBoolean(
				PARAMETER_MINIMIZE,
				"Indicates if to minimize or maximize the loss function.",
				true);
		type.setExpert(true);
		types.add(type);
		
		type = new ParameterTypeBoolean(
				PARAMETER_USE_LOCAL_RANDOM_SEED,
				"Indicates if to set the value of random seed.",
				false);
		type.setExpert(true);
		types.add(type);
		
		type = new ParameterTypeInt(
				PARAMETER_LOCAL_RANDOM_SEED,
				"The value of random seed",
				1, Integer.MAX_VALUE, 1992);
		
		type.setExpert(true);
		type.registerDependencyCondition(
				new BooleanParameterCondition(
						this, 
						PARAMETER_USE_LOCAL_RANDOM_SEED, 
						false,true));
		types.add(type);
		
		return types;
	}
	
	@Override
	public void doWork() throws OperatorException {
		
		start.deliver(new LayerSemaphore("0"));
		
		super.doWork();
		
		List<Operator> list = getSubprocess(0).getEnabledOperators();
		
		structure = convertStructure(getStructure(list));
		
		if (structure == null || structure.size() == 0){
			throw new OperatorException("Please specify the structure of the neural network "
					+ this.getName() +  ", at least one layer is needed");
		}
		
		// validate the input examples
		ExampleSet exampleSet = trainPort.getData(ExampleSet.class);

		// some checks
		if (exampleSet.getAttributes().getLabel() == null) {
			throw new UserError(this, 105);
		}
		if (exampleSet.getAttributes().size() == 0) {
			throw new UserError(this, 106);
		}
		if (exampleSet.size() == 0) {
			throw new UserError(this, 117);
		}
		
		// check capabilities and produce errors if they are not fulfilled
		CapabilityCheck check = new CapabilityCheck(this, com.rapidminer.tools.Tools.booleanValue(
				ParameterService.getParameterValue(
						PROPERTY_RAPIDMINER_GENERAL_CAPABILITIES_WARN), true)||onlyWarnForNonSufficientCapabilities());
		check.checkLearnerCapabilities(this, exampleSet);
		
		Model model = learn(exampleSet);
		modelPort.deliver(model);
		
		examplePort.deliver(exampleSet);
		
		// TODO for different class, the implementation is different.
	}
	
	abstract public Model learn(ExampleSet exampleSet) throws OperatorException;
	
	protected List<Operator> getStructure(List<Operator> list) throws OperatorException{
		
		List<Operator> result = new LinkedList<Operator>();
		
		for (Operator operator : list){
			if (operator.getClass() == SimpleOperatorChain.class){
				result.addAll(getStructure(((SimpleOperatorChain)operator).getSubprocess(0).getEnabledOperators()));
			} else if (AbstractLayer.class.isAssignableFrom(operator.getClass())){
				if (((AbstractLayer)operator).isLinked()){
					result.add(operator);
				}
				if (operator.getClass() == OutputLayer.class){
					return result;
				}
				
			} else {
				throw new OperatorException("Invalid operaoter nested in " + getName() +"; only layers allowed");
			}
		}
		
		return result;
	}
	
	protected List<AbstractLayer> convertStructure(List<Operator> list) throws OperatorException{
		List<AbstractLayer> result = new LinkedList<AbstractLayer>();
		for (int i=0;i<list.size();i++){
			result.add((AbstractLayer)list.get(i));
		}
		return result;
	}
	
	protected OptimizationAlgorithm getOptimizationAlgorithm(int i){
		switch (i) {
		case 0 : 
			return OptimizationAlgorithm.LINE_GRADIENT_DESCENT;
		case 1 : 
			return OptimizationAlgorithm.CONJUGATE_GRADIENT;
		case 2 : 
			return OptimizationAlgorithm.LBFGS;
		case 3 : 
			return OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
//		case 4 : 
//			return OptimizationAlgorithm.HESSIAN_FREE;
		default :
			return null;
		}	
	}
}
