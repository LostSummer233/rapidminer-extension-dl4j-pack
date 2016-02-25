package com.rapidminerchina.extension.dl4j.io;

import org.netlib.util.booleanW;

import com.rapidminer.operator.ResultObjectAdapter;
import com.rapidminer.tools.expression.internal.function.comparison.Equals;

public class LayerSemaphore extends ResultObjectAdapter {
	
	private String name = "";
	
	public LayerSemaphore(){
		this("");
	};
	
	public LayerSemaphore(String name){
		this.name = name;
	}
	
	public boolean equals(Object o){
		boolean b = false;
		if (o.getClass().equals(this.getClass())){
			b = true;
		}
		return b;
	}
	
	public String toString(){
		return "Semaphore " + name;
	}
}
