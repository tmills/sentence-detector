package org.apache.ctakes.core.cleartk.pipeline;

import java.io.FileWriter;
import java.io.IOException;

import org.apache.ctakes.core.ae.SimpleSegmentAnnotator;
import org.apache.ctakes.core.cleartk.ae.SentenceDetectorAnnotator;
import org.apache.ctakes.rnn.RnnSegmentDetector;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.resource.ResourceInitializationException;
import org.xml.sax.SAXException;

public class GenerateDescriptors {

  public static final String sentModelPath = "/org/apache/ctakes/core/sentdetect/model.jar";
  public static final String segModelPath = "/org/apache/ctakes/core/segdetect/model.jar";
  
  public static void main(String[] args) throws ResourceInitializationException, SAXException, IOException {
    AggregateBuilder aggregateBuilder = new AggregateBuilder();
    aggregateBuilder.add(SimpleSegmentAnnotator.createAnnotatorDescription());
    aggregateBuilder.add(SentenceDetectorAnnotator.getDescription(sentModelPath));
    
    aggregateBuilder.createAggregateDescription().toXML(new FileWriter("desc/analysis_engine/SentenceAnnotatorAggregate.xml"));
    SentenceDetectorAnnotator.getDescription(sentModelPath).toXML(new FileWriter("desc/analysis_engine/SentenceAnnotator.xml"));
    
    RnnSegmentDetector.getDescription(segModelPath).toXML(new FileWriter("desc/analysis_engine/SegmentAnnotator.xml"));
    
  }

}
