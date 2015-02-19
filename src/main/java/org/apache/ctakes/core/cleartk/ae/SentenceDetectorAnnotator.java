package org.apache.ctakes.core.cleartk.ae;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.log4j.Logger;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.DataWriter;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;
import org.cleartk.ml.jar.DefaultDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.util.ViewUriUtil;

public class SentenceDetectorAnnotator extends CleartkAnnotator<String>{

  private Logger logger = Logger.getLogger(SentenceDetectorAnnotator.class);

  @Override
  public void process(JCas jcas) throws AnalysisEngineProcessException {
    String uri = ViewUriUtil.getURI(jcas).toString();
    logger.info(String.format("Processing file with uri %s", uri));
    for(Segment seg : JCasUtil.select(jcas, Segment.class)){
      // keep track of next sentence during training
      List<Sentence> sents = JCasUtil.selectCovered(jcas, Sentence.class, seg);
      int sentInd = 0;
      Sentence nextSent = sents.size() > 0 ? sents.get(sentInd++) : null;
      int startInd=0;
      
      String prevOutcome = "<BEGIN>";
      String segText = seg.getCoveredText();
      for(int ind = 0; ind < segText.length(); ind++){
        List<Feature> feats = new ArrayList<>();
        
        feats.add(new Feature("PrevOutcome", prevOutcome));
        feats.add(new Feature("Character", segText.charAt(ind)));
        
        String outcome;
        if(this.isTraining()){
          // if ind pointer has passed nextSent pointer advance nextSent
          while(nextSent.getEnd() < ind && sentInd < sents.size()){
            nextSent = sents.get(sentInd++);
          }
          if(ind < nextSent.getBegin()){
            // current index is prior to next sentence
            outcome = "O";
          }else if(prevOutcome.equals("O")){
            // current index is in sentence but just after a character that was out of the sentence
            outcome = "B";
          }else{
            // current index is in the middle of a sentence
            outcome = "I";
          }
          this.dataWriter.write(new Instance<String>(outcome, feats));
        }else{
          outcome = this.classifier.classify(feats);
          if(outcome.equals("B")) startInd = ind;
          else if(outcome.equals("O") && 
              (prevOutcome.equals("I") || prevOutcome.equals("B"))){
            // just ended a sentence
            int endInd = ind-1;
            Sentence sent = new Sentence(jcas, startInd, endInd);
            sent.addToIndexes();            
          }
        }
        prevOutcome = outcome;
      }
    }
  }

  public static AnalysisEngineDescription getDataWriter(File outputDirectory, Class<? extends DataWriter<?>> class1) throws ResourceInitializationException {
    return AnalysisEngineFactory.createEngineDescription(
        SentenceDetectorAnnotator.class,
        SentenceDetectorAnnotator.PARAM_IS_TRAINING,
        true,
        DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
        outputDirectory,
        DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
        class1);
  }
}

