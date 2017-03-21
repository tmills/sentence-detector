package org.apache.ctakes.rnn;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.ctakes.core.cleartk.ae.SentenceDetectorAnnotator;
import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.log4j.Logger;
import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
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
import org.cleartk.ml.jar.GenericJarClassifierFactory;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class RnnSentenceDetector extends CleartkAnnotator<String>{
  public static final String PARAM_MODEL_FILE = "ModelFile";
  @ConfigurationParameter(name=PARAM_MODEL_FILE,mandatory=true)
  private File modelFile = null;
  private MultiLayerNetwork rnn = null;
  private Map<Character,Integer> charToIndexMap = null;
  private int inputSize;
  private Logger logger = Logger.getLogger(RnnSentenceDetector.class);
  
  @Override
  public void initialize(UimaContext arg0)
      throws ResourceInitializationException {
    super.initialize(arg0);
    
    ObjectInputStream ois;
    try {
      ois = new ObjectInputStream(new FileInputStream(modelFile));
      rnn = (MultiLayerNetwork) ois.readObject();
      ois.close();
    } catch (IOException | ClassNotFoundException e) {
      e.printStackTrace();
      throw new ResourceInitializationException(e);
    }
    charToIndexMap = new HashMap<>();
    char[] validChars = new LuceneReaderCharacterIterator().getCharacterSet();
    for( int i=0; i<validChars.length; i++ ) charToIndexMap.put(validChars[i], i);
    inputSize = validChars.length;
  }
  
  @Override
  public void process(JCas jcas) throws AnalysisEngineProcessException {


    for(Segment seg : JCasUtil.select(jcas, Segment.class)){
      rnn.rnnClearPreviousState();

      //Create input for initialization
      String initialization = " ";
      INDArray initializationInput = Nd4j.zeros(this.inputSize);
      char[] init = initialization.toCharArray();
      int idx = charToIndexMap.get(init[0]);
      initializationInput.putScalar(idx, 1.0f);
      rnn.rnnTimeStep(initializationInput);
//      output = output.tensorAlongDimension(output.size(2)-1,1,0); //Gets the last time step output

      // keep track of next sentence during training
      List<Sentence> sents = JCasUtil.selectCovered(jcas, Sentence.class, seg);
      int sentInd = 0;
      Sentence nextSent = sents.size() > 0 ? sents.get(sentInd++) : null;
      int startInd=0;
      
      String prevOutcome = "O";
      String segText = seg.getCoveredText();
      for(int ind = 0; ind < segText.length(); ind++){
        List<Feature> feats = new ArrayList<>();
        feats.add(new Feature("PrevOutcome", prevOutcome));
        
        char curChar = segText.charAt(ind);
        
        // put the character into the RNN and advance it one step:
        INDArray nextInput = Nd4j.zeros(this.inputSize);
        if(!charToIndexMap.containsKey(curChar)){
          System.err.println("Found unexpected char in input: " + curChar);
          curChar = ' ';
        }
        int curInd = charToIndexMap.get(curChar);
        nextInput.putScalar(curInd, 1.0f);    //Prepare next time step input
//        List<INDArray> activations = rnn.feedForward(nextInput);  //Do one time step of forward pass        
//        INDArray internal = activations.get(1);
        INDArray internal = rnn.rnnTimeStep(nextInput);
        INDArray prevState = (INDArray) ((BaseRecurrentLayer)rnn.getLayer(1)).rnnGetPreviousState().get("prevMem");
        for(int i = 0; i < internal.length(); i++){
          
          Double val = internal.getDouble(i);
          if(Double.isNaN(val)){
            val = 0.0;
          }
          feats.add(new Feature("ACT_COL_"+i, val));
        }
        
        for(int i = 0; i < prevState.length(); i++){
          Double val = prevState.getDouble(i);
          if(Double.isNaN(val)){
            val = 0.0;
          }
          feats.add(new Feature("MEM_IND_"+i, val));
        }
        
        // get the outcome and write/classify the example:
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
          if(!prevOutcome.equals("O") && Character.isLetterOrDigit(curChar)){
            outcome = "I";
          }else{
            outcome = this.classifier.classify(feats);
            if(outcome.equals("I") && prevOutcome.equals("O")){
              logger.warn("Classifier predicted I after an O -- setting to B instead to preserve BIO tagging structure.");
              outcome = "B";
            }else if(outcome.equals("B")){
              startInd = casInd;
            }else if(outcome.equals("O") && 
                (prevOutcome.equals("I") || prevOutcome.equals("B"))){
              // just ended a sentence
              int endInd = casInd;
              if(ind > 1){                  
                try{
                  while(endInd > startInd && Character.isWhitespace(segText.charAt(endInd-seg.getBegin()-1))){
                    endInd--;
                  }
                }catch(StringIndexOutOfBoundsException e){
                  System.err.println("Got an illegal index into this string!");
                }
              }
              if(endInd > startInd){
                SentenceDetectorAnnotator.makeSentence(jcas, startInd, endInd);
              }
            }
          }
        }
        prevOutcome = outcome;
 
      }
      if(!this.isTraining() && !prevOutcome.equals("O")){
        // segment ended with a sentence
        SentenceDetectorAnnotator.makeSentence(jcas, startInd, seg.getEnd());
      }
    }
    
  }
  
  public static AnalysisEngineDescription getDataWriter(File outputDirectory, Class<? extends DataWriter<?>> class1) throws ResourceInitializationException {
    return AnalysisEngineFactory.createEngineDescription(
        RnnSentenceDetector.class,
        RnnSentenceDetector.PARAM_IS_TRAINING,
        true,
        DirectoryDataWriterFactory.PARAM_OUTPUT_DIRECTORY,
        outputDirectory,
        DefaultDataWriterFactory.PARAM_DATA_WRITER_CLASS_NAME,
        class1,
        RnnSentenceDetector.PARAM_MODEL_FILE,
        "rnn_model.obj");
  }

  
  public static AnalysisEngineDescription getDescription(String modelPath) throws ResourceInitializationException {
    return AnalysisEngineFactory.createEngineDescription(
        RnnSentenceDetector.class,
        RnnSentenceDetector.PARAM_IS_TRAINING,
        false,
        GenericJarClassifierFactory.PARAM_CLASSIFIER_JAR_PATH,
        modelPath,
        RnnSentenceDetector.PARAM_MODEL_FILE,
        "rnn_model.obj");
  }
}
