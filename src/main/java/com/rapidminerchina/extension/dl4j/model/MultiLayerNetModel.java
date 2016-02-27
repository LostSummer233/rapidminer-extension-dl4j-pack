package com.rapidminerchina.extension.dl4j.model;

import java.util.ArrayList;
import java.util.List;

import org.apache.poi.poifs.storage.ListManagedBlock;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.deeplearning4j.nn.layers.feedforward.rbm.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Attributes;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.set.ExampleSetUtilities;
import com.rapidminer.example.table.NominalMapping;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.learner.*;
import com.rapidminerchina.extension.dl4j.DL4JConvert;



/**
 * A multilayer network model.
 * This model functions as an interface between RM and DL4J;
 * it contains all information about the miltolayer network used in DL4J,
 * information may be duplicated for fast access.
 * 
 * This class is still under development as some functions are not completely implemented,
 * with the current functions, the extension is able to work on certtain tasks.
 * 
 * The information contained in this model are:
 * <ul>
 * 	<li>The multilayer network model itself;</li>
 * 	<li>The name of features;</li>
 * 	<li>The mean and standard deviation of the training data;</li>
 * 	<li>The configuration information of the multilayer network model;</li>
 * 	<li>The transfer matrices (weights) between each two layers</li>
 * </ul>
 * 
 * Currently:
 * <ul>
 * 	<li>The transfer matrices are not implemented, as they are not used;</li>
 * 	<li>The method clone is not implemented, as they are not used in current implementation.</li>
 * </ul>
 * 
 * @author Anson Chen
 * @version 0.3.1
 */
@SuppressWarnings("serial")
public class MultiLayerNetModel extends PredictionModel{
	
	/**
	 * The list of name of each layer.
	 */
	private List<String> names;
	
	/**
	 * The multilayer network model.
	 */
	private MultiLayerNetwork model;
	
	/**
	 * The list stores feature names.
	 */
	List<String> featureNames;
	
	/**
	 * The means of features of the training data, typically used for normalization.
	 */
	private INDArray columnMeans = null;
	
	/**
	 * The standard deviation of features of the training data, typically used for normalization.
	 */
	private INDArray columnStds = null;

	/**
	 * The configuration information of the mutilayer network model.
	 */
	private MultiLayerConfiguration configuration;
	
	/*
	 * The transfer matrices (weights and bias) represented in 2d-arrays.
	 * Designed as a quick alternative to access the weights and bias than access it via model.
	 * Currently not used in the implementation, thus not implemented.
	 */
//	private List<INDArray> transferMatrices;
	
	/**
	 * Constructor.
	 * 
	 * @param exampleSet the training exampleset
	 * 
	 * Notice this constructor does not train a model on the training examples, 
	 * but simply record the feature names of the training examples.
	 */
    public MultiLayerNetModel(ExampleSet exampleSet) {
        super(exampleSet,ExampleSetUtilities.SetsCompareOption.ALLOW_SUPERSET, ExampleSetUtilities.TypesCompareOption.ALLOW_SAME_PARENTS);
		
        // record the attributes
        Attributes attributes = exampleSet.getAttributes();
		featureNames = new ArrayList<String>();
		for (Attribute attribute : attributes){
			featureNames.add(attribute.getName());
		}
		
		model = null;
        configuration = null;
    }

    /**
     * Retrieve the names of the features.
     * @return a list of the feature names
     */
    public List<String> getFeaturName(){
    	return featureNames;
    }
    
    /**
     * Retrieve the name of the label attribute.
     * Notice it is NOT the names of the nominal values of the labels. 
     * @return the name of the label attribute
     */
    public Attribute getLabel(){
    	return getTrainingHeader().getAttributes().getLabel();
    }
    
    /**
     * Retrieve the nominal mapping that maps the label names to numerics.
     * @return the nominal mapping
     */
    public NominalMapping getLabelMapping(){
    	return getLabel().getMapping();
    }
    
    /**
     * Retrieve the names of labels, i.e. the possible values of the label attributes.
     * @return a list of the label names
     */
    public List<String> getLabelName(){
    	return getLabelMapping().getValues();
    }
    
    /**
     * Retrieve the trained multilayer network model.
     * @return the multilayer network model
     */
    public MultiLayerNetwork getModel(){
    	return model;
    }
    
    /**
     * Retrieve the configuration of the multilayer network model.
     * @return the configuration of the multilayer network model
     */
    public MultiLayerConfiguration getConfiguration(){
    	return configuration;
    }
    
//    /**
//     * Retrieve the weights (including bias) between each two layers,
//     * each set of weights is represented as a 2d-array.
//     * @return a list of the weights between each two layers
//     */
//    public List<INDArray> getTransferMatrices(){
//    	return transferMatrices;
//    }
    
//    /**
//     * Retrieve the weights (including bias) between two layers specified by an index i. 
//     * @param i the index of the transfer matrix
//     * @return the specified transfer matrix
//     */
//    public INDArray getTransferMatrix(int i){
//    	return transferMatrices.get(i);			
//    }
    
    /**
     * Overwrite the names of the features of this model.
     * @param names the names of the features to specify
     */
    public void setFeatureNames(List<String> names){
    	this.featureNames = names;
    }
    
    /**
     * Overwrite the model.
     * @param model the model
     */
    public void setMultiLayerNetwork(MultiLayerNetwork model){
    	this.model = model;
    	this.configuration = model.getLayerWiseConfigurations();
    }
    
    /**
     * Overwrite the configuration of this model
     * @param configuration the configuration
     */
    public void setMultiLayerConfiguration(MultiLayerConfiguration configuration){
    	this.configuration = configuration;
    }
    
    /**
     * Perform prediction and write the results to a specified attribute name.
     */
	@Override
	public ExampleSet performPrediction(ExampleSet exampleSet, Attribute predictedLabel) throws OperatorException {
		
		if (exampleSet.getAttributes().getPredictedLabel() != predictedLabel){
			exampleSet.getAttributes().setPredictedLabel(predictedLabel);
		}
		
		// construct the table of features
		int feature_num = featureNames.size();
		int row_num = exampleSet.size();
		double[][] featuresMatrix = new double[row_num][feature_num];
		
		int counter = 0;
		for (Example e : exampleSet){
			for (int i=0; i<feature_num; i++){
				double d = e.getValue(exampleSet.getAttributes().get(featureNames.get(i)));
				featuresMatrix[counter][i] = d;
			}
			counter++;
		}
		
		// build the 2d-array of the features
		INDArray features = org.nd4j.linalg.factory.Nd4j.create(featuresMatrix);
		
		// normalize features in the same way that the training data is normalized.
		if (this.columnMeans != null && this.columnStds != null){
			features = features.subiRowVector(columnMeans);
			features = features.diviRowVector(columnStds);
		}
		
		// make prediction
		INDArray output = model.output(features);
		
		/*
		 * convert the output 2d-array to the specified attribute in the exampleset,
		 * together with the confidences on each label
		 */
		int[] indices = DL4JConvert.getMax(output);
		NominalMapping mapping = getLabelMapping();
		int numLabel = mapping.getValues().size();
		
		counter = 0;
		for (Example e : exampleSet){
			
			e.setPredictedLabel(indices[counter]);
			
			for (int i=0; i<numLabel; i++){
				e.setConfidence(mapping.mapIndex(i), output.getRow(counter).getDouble(i));
			}
			counter++;
		}
		
		return exampleSet;
	}
	
	/**
	 * Train the data.
	 * 
	 * @param exampleSet the training exampleset
	 * @param shuffle whether to shuffle the examples;
	 *        I may not include this boolean in later version as RM has its own shuffle method in split operator
	 * @param normalization whether to normalize each column
	 */
	public void train(ExampleSet exampleSet, boolean shuffle, boolean normalization){
		
		this.model = new MultiLayerNetwork(configuration);
		model.init();
		DataSet data = DL4JConvert.convert2DataSet(exampleSet);

		/*
		 * Haven't check version 3.8, but in version 3.7, shuffle() is not correctly implemented
		 * in DL4J, so currently we shuffle the examples before they enter learners.
		 */
		
//		if (shuffle){
//			data.shuffle();
//		}
		
		// normalize the training data and record the mean and standard deviation
		if (normalization){
			this.columnMeans = data.getFeatures().mean(0);
			this.columnStds = data.getFeatureMatrix().std(0);
	        data.setFeatures(data.getFeatures().subiRowVector(columnMeans));
	        this.columnStds.addi(Nd4j.scalar(Nd4j.EPS_THRESHOLD));
	        data.setFeatures(data.getFeatures().diviRowVector(columnStds));
		} else {
			this.columnMeans = org.nd4j.linalg.factory.Nd4j.zeros(data.getFeatures().columns());
			this.columnStds = org.nd4j.linalg.factory.Nd4j.ones(data.getFeatures().columns());
		}
		
		// train the model
		model.fit(data);
	}
	
	/**
	 * Train the multilayer network model with a refined configurations
	 * This method is only used if the configuration of the model is not defined (as null) when the model is constructed.
	 * This method may be deprecated later
	 */
	public void train(ExampleSet exampleSet, MultiLayerConfiguration configuration, 
			boolean shuffle, boolean normalization, List<String> layerNames){
		this.configuration = configuration;
		this.names = layerNames;
		train(exampleSet, shuffle, normalization);
	}
	
	/**
	 * Clone the model with full information
	 * This method is not used implemented and may be deprecated later.
	 * Not used in current implementation, thus not implemented.
	 */
	public MultiLayerNetModel clone(){
		return null;
	}
	
	
	public String toString(){
		
		String result = "Neural Network:";
		result += "\n ===================================\n";
		
		int numLayer = model.getnLayers();
		
		configuration.toString();
		
		for (int i=0; i<numLayer; i++){
			org.deeplearning4j.nn.api.Layer layer = model.getLayer(i);
			// don't know why there is no API like getActivationFunction()......
//			String activation = model.getLayer(i)
			
//			result += "\n" + layer.paramTable().get("W").toString();
//			result += "\n";
			
			
			
			if (layer.getClass() == DenseLayer.class){
				
				result += "\nLayer " + i + ": " + names.get(i) + " (Dense Layer)" ;
				result += "\n ------------------------------------";
				INDArray array = layer.paramTable().get("W");
				for (int j=0; j<array.shape()[0]; j++){
					result += "\n" + array.getRow(j).toString();
				}
				result += "\n" + model.getLayer(i).paramTable().get("b");
				result += "\n";
				
			} else if (layer.getClass() == OutputLayer.class){
				
				result += "\nLayer " + i + ": " + names.get(i) + " (Output Layer)";
				result += "\n ------------------------------------";
				INDArray array = layer.paramTable().get("W");
				for (int j=0; j<array.shape()[0]; j++){
					result += "\n" + array.getRow(j).toString();
				}
				result += "\n" + model.getLayer(i).paramTable().get("b");
				result += "\n";
				
			} else if (layer.getClass() == RBM.class){
				
				result += "\nLayer " + i + ": " + names.get(i) + " (RBM Layer)";
				result += "\n ------------------------------------";
				INDArray array = layer.paramTable().get("W");
				for (int j=0; j<array.shape()[0]; j++){
					result += "\n" + array.getRow(j).toString();
				}
				result += "\n" + model.getLayer(i).paramTable().get("b");
				result += "\n";
				
			} else if (layer.getClass() == SubsamplingLayer.class){
				
				result += "\nLayer " + i + ": " + names.get(i) + " (Subsampling Layer)";
				result += "\n ------------------------------------";
				result += "\n";
				
			} else if (layer.getClass() == ConvolutionLayer.class){
				
				result += "\nLayer " + i + ": " + names.get(i) + " (Convolutional Layer)";
				result += "\n ------------------------------------";
				INDArray array3D = layer.paramTable().get("W");
				
//				int depth = array3D.shape()[2];
				
				result += "\n" + array3D.toString();
				
//				for (int d = 0; d<depth; d++){
//					result += "\n    Filter " + d;
//					INDArray array2D = array3D.tensorssAlongDimension(dimension);
//					
//					for (int j=0; j<array2D.shape()[0]; j++){
//						result += "\n" + array2D.getRow(j).toString();
//					}
//
//					result += "\n" + model.getLayer(i).paramTable().get("b").getRow(d);					
//				}
				
				result += "\n";
				
			}
		}
		
		return result;
	}
	
}
