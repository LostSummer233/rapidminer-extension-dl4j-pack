package com.rapidminerchina.extension.dl4j.word2vec;

import org.deeplearning4j.models.word2vec.Word2Vec;

import com.rapidminer.example.ExampleSet;
import com.rapidminer.operator.AbstractModel;
import com.rapidminer.operator.OperatorException;

public class Word2VecModel extends AbstractModel {

	private Word2Vec vec;
	private ExampleSet resultTable;
	
	protected Word2VecModel(ExampleSet exampleSet) {
		super(exampleSet);
		// TODO Auto-generated constructor stub
	}
	
	public void setModel(Word2Vec vec){
		this.vec = vec;
	}
	
	public void setResultTable(ExampleSet result){
		this.resultTable = result;
	}
	
	public Word2Vec getModel(){
		return this.vec;
	}
	
	public ExampleSet getResult(){
		return this.resultTable;
	}

	@Override
	public ExampleSet apply(ExampleSet testSet) throws OperatorException {
		// TODO Auto-generated method stub
		return null;
	}
	
	

}
