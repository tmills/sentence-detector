package org.apache.ctakes.core.cleartk.ae;

import java.io.File;
import java.io.IOException;

import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.util.ViewUriUtil;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;

public class AnaforaSentenceXmlReader extends JCasAnnotator_ImplBase {

  public static AnalysisEngineDescription getDescription() throws ResourceInitializationException{
    return AnalysisEngineFactory.createEngineDescription(AnaforaSentenceXmlReader.class);
  }
  
  @Override
  public void process(JCas jcas) throws AnalysisEngineProcessException {
    File txtFile = new File(ViewUriUtil.getURI(jcas));
    String xmlUri = txtFile.getAbsolutePath() + ".Sentences.tim.completed.xml";
    
    Element dataElem;
    try {
      dataElem = new SAXBuilder().build(xmlUri).getRootElement();
    } catch (JDOMException | IOException e) {
      e.printStackTrace();
      throw new AnalysisEngineProcessException(e);
    }
    
    for (Element annotationsElem : dataElem.getChildren("annotations")) {
    
      String[] span = null;
      String type = null;
      for (Element entityElem : annotationsElem.getChildren("entity")) {
        Element spanElem = entityElem.getChild("span");
        span = spanElem.getText().split(",");
        type = entityElem.getChild("type").getText();     
      

        int begin = Integer.parseInt(span[0]);
        int end = Integer.parseInt(span[span.length-1]);

        if(type.equals("Segment")){
          Segment segment = new Segment(jcas, begin, end);
          segment.addToIndexes();
        }else if(type.equals("Sentence")){
          Sentence sent = new Sentence(jcas, begin, end);
          sent.addToIndexes();
        }
      }
    }
  }

}
