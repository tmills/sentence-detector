package org.apache.ctakes.core.cleartk.pipeline;

import java.io.FileWriter;
import java.io.IOException;

import org.apache.ctakes.core.ae.SimpleSegmentAnnotator;
import org.apache.ctakes.core.cleartk.ae.SentenceDetectorAnnotator;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.resource.ResourceInitializationException;
import org.xml.sax.SAXException;

public class GenerateDescriptors {

  public static final String modelPath = "/org/apache/ctakes/core/sentdetect/model.jar";
  
  public static void main(String[] args) throws ResourceInitializationException, SAXException, IOException {
    AggregateBuilder aggregateBuilder = new AggregateBuilder();
    aggregateBuilder.add(SimpleSegmentAnnotator.createAnnotatorDescription());
    aggregateBuilder.add(SentenceDetectorAnnotator.getDescription(modelPath));
    
    aggregateBuilder.createAggregateDescription().toXML(new FileWriter("desc/analysis_engine/SentenceAnnotatorAggregate.xml"));
    SentenceDetectorAnnotator.getDescription(modelPath).toXML(new FileWriter("desc/analysis_engine/SentenceAnnotator.xml"));
  }

}
