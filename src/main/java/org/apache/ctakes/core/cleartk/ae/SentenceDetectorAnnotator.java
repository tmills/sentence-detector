package org.apache.ctakes.core.cleartk.ae;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.log4j.Logger;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CASRuntimeException;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkAnnotator;
import org.cleartk.ml.DataWriter;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;
import org.cleartk.ml.feature.function.CharacterCategoryPatternFunction;
import org.cleartk.ml.feature.function.CharacterCategoryPatternFunction.PatternType;
import org.cleartk.ml.jar.DefaultDataWriterFactory;
import org.cleartk.ml.jar.DirectoryDataWriterFactory;
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.cleartk.util.ViewUriUtil;

public class SentenceDetectorAnnotator extends CleartkAnnotator<String>{

  private Logger logger = Logger.getLogger(SentenceDetectorAnnotator.class);
  private static final int WINDOW_SIZE = 3;
  
  @Override
  public void process(JCas jcas) throws AnalysisEngineProcessException {
    try{
      String uri = ViewUriUtil.getURI(jcas).toString();
      logger.info(String.format("Processing file with uri %s", uri));
    }catch(CASRuntimeException e){
      logger.debug("No uri found, probably not a big deal unless this is an evaluation.");
    }
    
    for(Segment seg : JCasUtil.select(jcas, Segment.class)){
      // keep track of next sentence during training
      List<Sentence> sents = JCasUtil.selectCovered(jcas, Sentence.class, seg);
      int sentInd = 0;
      Sentence nextSent = sents.size() > 0 ? sents.get(sentInd++) : null;
      int startInd=0;
      
      String prevOutcome = "O";
      String segText = seg.getCoveredText();
      for(int ind = 0; ind < segText.length(); ind++){
        List<Feature> feats = new ArrayList<>();
        
        char curChar = segText.charAt(ind);
        
        feats.add(new Feature("PrevOutcome", prevOutcome));
        feats.addAll(getCharFeatures(curChar, "Character"));

        for(int window = -WINDOW_SIZE; window <= WINDOW_SIZE; window++){
          if(ind+window >= 0 && ind+window < segText.length()){
            char conChar = segText.charAt(ind+window);
            feats.addAll(getCharFeatures(conChar, "CharOffset_"+window));
          }
        }
        
        String nextToken = getNextToken(segText, ind);
        String prevToken = getPrevToken(segText, ind);
        feats.addAll(getTokenFeatures(nextToken, "Next"));
        feats.addAll(getTokenFeatures(prevToken, "Prev"));
        
        String outcome;
        int casInd = seg.getBegin() + ind;
        if(this.isTraining()){
          // if ind pointer has passed nextSent pointer advance nextSent
          while(nextSent != null && nextSent.getEnd() < casInd && sentInd < sents.size()){
            nextSent = sents.get(sentInd++);
          }
          if(nextSent == null){
            outcome = "O";
          }else if(casInd < nextSent.getBegin()){
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
          if(outcome.equals("B")) startInd = casInd;
          else if(outcome.equals("O") && 
              (prevOutcome.equals("I") || prevOutcome.equals("B"))){
            // just ended a sentence
            int endInd = casInd-1;
            while(endInd > startInd && Character.isWhitespace(segText.charAt(endInd-seg.getBegin()))){
              endInd--;
            }
            
            if(endInd > startInd){
              Sentence sent = new Sentence(jcas, startInd, endInd);
              sent.addToIndexes();    
            }
          }
        }
        prevOutcome = outcome;
      }
      if(!this.isTraining() && !prevOutcome.equals("O")){
        // segment ended with a sentence
        Sentence sent = new Sentence(jcas, startInd, seg.getEnd());
        sent.addToIndexes();
      }
    }
  }

  private static String getNextToken(String segText, int ind) {
    int startInd = ind;
    
    // move startInd right if it's whitespace and left if it's not.
    while(startInd < segText.length() && Character.isWhitespace(segText.charAt(startInd))){
      startInd++;
    }
    while(startInd > 0 && !Character.isWhitespace(segText.charAt(startInd-1))){
      startInd--;
    }
    
    int endInd = startInd;
    while(endInd < segText.length() && !Character.isWhitespace(segText.charAt(endInd))){
      endInd++;
    }
    
    return segText.substring(startInd, endInd);    
  }
  
  private static String getPrevToken(String segText, int ind){
    int endInd = ind;
    
    // move endInd left until we hit whitespace:
    while(endInd > 0 && !Character.isWhitespace(segText.charAt(endInd))){
      endInd--;
    }
    // then move until the character to the left is whitespace
    while(endInd > 0 && Character.isWhitespace(segText.charAt(endInd))){
      endInd--;
    }
    
    int startInd = endInd;
    while(startInd > 0 && !Character.isWhitespace(segText.charAt(startInd)) && !Character.isWhitespace(segText.charAt(startInd-1))){
      startInd--;
    }
    
    return segText.substring(startInd, endInd+1);
  }

  static CharacterCategoryPatternFunction<Annotation> shapeFun = new CharacterCategoryPatternFunction<>(PatternType.REPEATS_AS_KLEENE_PLUS);
  
  private static Collection<? extends Feature> getTokenFeatures(String token, String prefix) {
    List<Feature> feats = new ArrayList<>();
    
    Feature tokenFeat = new Feature(prefix + "TokenIdentity", token);
    feats.add(tokenFeat);
    feats.add(new Feature(prefix+"length="+token.length(), true));
    feats.add(new Feature(prefix+"cap", token.length() > 0 && Character.isUpperCase(token.charAt(0))));
    feats.addAll(shapeFun.apply(tokenFeat));
    return feats;
  }

  public static List<Feature> getCharFeatures(char ch, String prefix){
    List<Feature> feats = new ArrayList<>();
    feats.add(new Feature(prefix+"_Type", ch == '\n' ? "<LF>" : ch));
    feats.add(new Feature(prefix+"_Upper", Character.isUpperCase(ch)));
    feats.add(new Feature(prefix+"_Lower", Character.isLowerCase(ch)));
    feats.add(new Feature(prefix+"_Digit", Character.isDigit(ch)));
    feats.add(new Feature(prefix+"_Space", Character.isWhitespace(ch)));
    feats.add(new Feature(prefix+"_Type"+Character.getType(ch), true));
    return feats;
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

  public static AnalysisEngineDescription getDescription(String modelPath) throws ResourceInitializationException {
    return AnalysisEngineFactory.createEngineDescription(
        SentenceDetectorAnnotator.class,
        SentenceDetectorAnnotator.PARAM_IS_TRAINING,
        false,
        GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
        modelPath);
  }
}

