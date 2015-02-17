package org.apache.ctakes.core.cleartk.ae;

import java.util.ArrayList;
import java.util.List;

import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;

public class SentenceDetectorAnnotator extends CleartkAnnotator<String>{

  @Override
  public void process(JCas jcas) throws AnalysisEngineProcessException {
    for(Segment seg : JCasUtil.select(jcas, Segment.class)){
      // keep track of next sentence during training
      List<Sentence> sents = JCasUtil.selectCovered(jcas, Sentence.class, seg);
      Sentence nextSent = sents.size() > 0 ? sents.remove(0) : null;
      int startInd=0;
      
      String prevOutcome = "<BEGIN>";
      String segText = seg.getCoveredText();
      for(int ind = seg.getBegin(); ind <= seg.getEnd(); ind++){
        List<Feature> feats = new ArrayList<>();
        
        feats.add(new Feature("PrevOutcome", prevOutcome));
        feats.add(new Feature("Character", segText.charAt(ind)));
        
        String outcome;
        if(this.isTraining()){
          // if ind pointer has passed nextSent pointer advance nextSent
          while(nextSent.getEnd() < ind){
            nextSent = sents.remove(0);
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
}

