package com.rapidminerchina.extension.dl4j.layers;

import com.fasterxml.jackson.core.sym.Name;
import com.rapidminer.operator.Operator;
import com.rapidminer.operator.OperatorDescription;
import com.rapidminer.operator.OperatorException;
import com.rapidminer.operator.UserError;
import com.rapidminer.operator.ports.InputPort;
import com.rapidminer.operator.ports.OutputPort;
import com.rapidminer.operator.ports.metadata.PassThroughRule;
import com.rapidminer.parameter.UndefinedParameterError;
import com.rapidminerchina.extension.dl4j.io.LayerSemaphore;

import org.deeplearning4j.nn.conf.layers.Layer;

public abstract class AbstractLayer extends Operator{

	
	private final InputPort inPort = getInputPorts().createPort("through");
	private final OutputPort outPort = getOutputPorts().createPort("through");
	
	public AbstractLayer(OperatorDescription description) {
		super(description);
		getTransformer().addRule(new PassThroughRule(inPort, outPort, false));
		getTransformer().addGenerationRule(outPort, LayerSemaphore.class);
	}
	
	public abstract Layer getLayer() throws UndefinedParameterError;
	
	public abstract Layer getLayer(int i) throws UndefinedParameterError;
	
	public boolean isLinked() throws UserError {
		
		LayerSemaphore semaphore = inPort.getDataOrNull(LayerSemaphore.class);
		if (semaphore != null && semaphore.getClass() == LayerSemaphore.class){
			return true;
		} else {
			return false;
		}
		
	}
	
	@Override
	public void doWork() throws OperatorException {
		super.doWork();
		outPort.deliver(inPort.getDataOrNull(LayerSemaphore.class));
	}
	
	public abstract int getNumNodes() throws UndefinedParameterError;
	
	public abstract String getLayerName();
	
}