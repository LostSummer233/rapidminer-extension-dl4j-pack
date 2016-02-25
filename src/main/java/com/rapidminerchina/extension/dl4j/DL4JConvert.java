package com.rapidminerchina.extension.dl4j;

import com.rapidminer.example.Attribute;
import com.rapidminer.example.Attributes;
import com.rapidminer.example.Example;
import com.rapidminer.example.ExampleSet;
import com.rapidminer.example.ExampleSetFactory;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * The utility functions that convert data structures between RM and DL4J.
 * 
 * This class is under development, functions may be add or removed during the process.
 * Currently, it contains method(s) that:
 * <ul>
 * 	<li> converts an RM exampleset into a DL4J dataset;</li>
 * 	<li> converts a DL4J dataset into an RM exampleset;</li>
 * 	<li> retrieves the index of the maximum value on each row of an 2d-array,
 * 		(typically used in get the indices of max confidence);</li>
 * </ul>
 * 
 * @author Anson Chen
 * @version 0.3
 */

public class DL4JConvert {
	
	/**
	 * Convert an exampleset into a dataset that can be used in DL4J 
	 * @param exampleSet the exampleset to convert
	 * @param convertLabel whether to convert the label column in the exampleset if there is one
	 * <ul>
	 * 	<li> {@value true} the generated dataset contains a table (a 2d-array) of labels when applicable </li>
	 * 	<li> {@value false} the generated dataset contains an empty table (a 2d-array) of labels </li>
	 * </ul>
	 * @return the generated dataset
	 */
	public static DataSet convert2DataSet(ExampleSet exampleSet, boolean convertLabel){
		
		int row_num = exampleSet.size();
		
		// construct the list of feature names
		Attributes attributes = (Attributes) exampleSet.getAttributes();
		List<String> featureNames = new ArrayList<String>();
		for (Attribute attribute : attributes){
			featureNames.add(attribute.getName());
		}
		
		int feature_num = featureNames.size();
		
		// construct the list of labels
		Attribute labelAttribute = (Attribute) exampleSet.getAttributes().getLabel();
		boolean toConvertLabel = (labelAttribute != null) && convertLabel;
		
		List<String> labelNames = new ArrayList<String>();
		if (toConvertLabel){
			// the list is supposed to be in order, or at least the indices of all values are continuous natural numbers form 0
			labelNames = (List<String>)(labelAttribute.getMapping().getValues());
		}
		
		int label_num = labelNames.size();

		// construct the table of features and labels
		double[][] featuresMatrix = new double[row_num][feature_num];
		double[][] labelsMatrix = new double[row_num][label_num];
		
		int counter = 0;
		
		for (Example e : exampleSet){
			for (int i=0; i<feature_num; i++){
				double d = e.getValue(attributes.get(featureNames.get(i)));
				featuresMatrix[counter][i] = d;
			}
			if (toConvertLabel){
				int l = (int) e.getLabel();
				labelsMatrix[counter][l] = 1;
			}

			counter++;
		}
		
		// build the 2d-arrays for features and labels
		INDArray features = org.nd4j.linalg.factory.Nd4j.create(featuresMatrix);
		INDArray labels = org.nd4j.linalg.factory.Nd4j.create(labelsMatrix);
		
		// construct the dataset based on the features and labels, then add names to features and labels 
		DataSet dataSet = new DataSet(features, labels);
		dataSet.setColumnNames(featureNames);
		dataSet.setLabelNames(labelNames);
		return dataSet;
	}
	
	/**
	 * A shortcut for convert2DataSet(exampleSet, true), refer {@link convert2DataSet}
	 * @return the generated dataset
	 */
	public static DataSet convert2DataSet(ExampleSet exampleSet){
		return convert2DataSet(exampleSet, true);
	}
	
	/**
	 * Convert a dataset together with an array of prediction into ExampleSet.
	 * 
	 * This method is implemented before I get a deep look at the learner classes in RM,
	 * thus it is improperly implemented, and is never invoked.
	 * I may remodify this method for other usage.
	 * 
	 * @param dataset the given dataset
	 * @param predict the predicted results
	 * @param predictLabelNames the list of names mapping the predicted result to a nominal class/label
	 * @param convertLabel whether to convert the true label column recorded in dataset
	 * @param convertConfidence whether to add the confidence as a column in the resulting exampleset 
	 * @return the generated exampleset
	 */
	
	@ Deprecated
	public static ExampleSet convert2ExampleSet(org.nd4j.linalg.dataset.api.DataSet dataSet, INDArray predict, List<String> predictLabelNames, boolean convertLabel, boolean convertConfidence){
		
		// check if label is to be converted
		boolean toConvertLabel = convertLabel;
		List<String> labelNames = dataSet.getLabelNames();
		INDArray labels = dataSet.getLabels();
		
		if (labelNames == null || labels == null){
			toConvertLabel = false;
		} else if (labelNames.size() == 0 || labels.shape()[0] == 0 || labels.shape()[1] == 0){
			toConvertLabel = false;
		}
		
		// construct the matrix of data used in the exampleset
		INDArray features = dataSet.getFeatures();
		List<String> featureNames = dataSet.getColumnNames();
		int row_num = features.shape()[0];
		int fea_num = features.shape()[1];
		int col_num = fea_num + 1;
		
		// get the indices of the labels and/or predicted labels
		int[] indices = new int[0];
		
		if (toConvertLabel){
			col_num ++;
			indices = getMax(labels);
		}
		
		if (convertConfidence){
			col_num ++;
		}

		Object[][] data = new Object[row_num][col_num];
		int[] predictIndices = getMax(predict);
		
		// fill the data matrix
		for (int i=0; i<row_num; i++){
			for (int j=0; j< fea_num; j++){
				data[i][j] = features.getRow(i).getDouble(j);
			}
			
			if (toConvertLabel){
				data[i][fea_num] = labelNames.get(indices[i]);
				data[i][fea_num +1] = labelNames.get(predictIndices[i]);
			} else {
				data[i][fea_num] = labelNames.get(predictIndices[i]);
			}
			
			if(convertConfidence){
				data[i][col_num-1] = predict.getRow(i).getDouble(predictIndices[i]);
			}
		}
		
		// construct the sxampleset and the attributes
		ExampleSet result = ExampleSetFactory.createExampleSet(data);
		
		Attributes attributes = result.getAttributes();
		
		// set the feature names
		for (int j=1; j < fea_num + 1; j++){
			
			// a resolution for duplicated names
			int suffix = 0;
			String attributeName = featureNames.get(j-1);
			while (attributes.get(attributeName) != null){
				attributeName = featureNames.get(j-1) + "("+ suffix +")";
				suffix ++;
			}
			
			attributes.get("att"+j).setName(attributeName);
		}
		
		// set the special attributes' names and roles
		if (toConvertLabel){
			attributes.get("att" + (fea_num+1)).setName("Label");
			attributes.setLabel(attributes.get("Label"));
			
			attributes.get("att" + (fea_num+2)).setName("Prediction");
			attributes.setPredictedLabel(attributes.get("Prediction"));
		} else {
			attributes.get("att" + (fea_num+1)).setName("Prediction");
			attributes.setPredictedLabel(attributes.get("Prediction"));
		}
		
		// add confidence to the exampleset
		if (convertConfidence){
			attributes.get("att" + (col_num)).setName("Confidence");
			attributes.setSpecialAttribute(attributes.get("Confidence"), "Confidence");
		}

		return result;
		
	}
	
	/**
	 * A shortcut of convert2ExampleSet(dataSet, predict, labelNames, true, true);
	 * 
	 * Also deprecated.
	 * 
	 * @return the generated exampleset
	 */
	
	@ Deprecated
	public static ExampleSet convert2ExampleSet(org.nd4j.linalg.dataset.api.DataSet dataSet, INDArray predict, List<String> labelNames){
		
		return convert2ExampleSet(dataSet, predict, labelNames, true, true);
	}
	
	/**
	 * A support function to retrieve the indices of the max value on each row of a 2d-array.
	 * This method is particularly used to retrieve the indices of predicted labels from a matrix of confidences.
	 * 
	 * @param array the 2d-array
	 * @return an array of indices indicating the max value on each row of the 2d-array
	 */
	public static int[] getMax(INDArray array){
		
		int[] shape = array.shape();
		int[] maxList = new int[shape[0]]; 
		
		for (int i=0; i<shape[0]; i++){
			INDArray row = array.getRow(i);
			double max = row.getDouble(0);
			for (int j=1; j<shape[1]; j++){
				if (row.getDouble(j) > max){
					maxList[i] = j;
					max = row.getDouble(j);
				}
			}
		}
		return maxList;
	}
}


