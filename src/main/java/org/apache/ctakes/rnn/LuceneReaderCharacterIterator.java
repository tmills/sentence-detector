package org.apache.ctakes.rnn;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexableField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public class LuceneReaderCharacterIterator implements DataSetIterator {

  /**
   * 
   */
  private static final long serialVersionUID = 1L;
  
  public static final int MINI_BATCH_SIZE = 50;
  public static final int EXAMPLE_SIZE = 100;
  public static final String textFieldName = "content";
  
  private DirectoryReader ireader = null;
  private int charNum = 0;
  private int maxChars = -1;  // the shakespeare corpus is roughly 6 million chars according to wc -c
  private char[] validChars = null;
  private int numCharacters = 0;
  private Random rand = new Random(718);
  private Map<Character,Integer> charToIdxMap = null;
  
  public LuceneReaderCharacterIterator(String indexDir){
    Directory dir;
    this.maxChars = MINI_BATCH_SIZE * 500;
    this.validChars = getMimicCharacterSet();
    this.numCharacters = validChars.length;
    //Store valid characters is a map for later use in vectorization
    charToIdxMap = new HashMap<>();
    for( int i=0; i<validChars.length; i++ ) charToIdxMap.put(validChars[i], i);
    
    try {
      dir = FSDirectory.open(new File(indexDir));
      ireader = DirectoryReader.open(dir);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
  @Override
  public boolean hasNext() {
    return (charNum + MINI_BATCH_SIZE) < maxChars;
  }

  @Override
  public DataSet next() {
    return next(MINI_BATCH_SIZE);
  }

  @Override
  public int batch() {
    return MINI_BATCH_SIZE; // copied from RNN example
  }

  @Override
  public int cursor() {
    return charNum;
  }

  @Override
  public int inputColumns() {
    return this.validChars.length;
  }

  @Override
  public DataSet next(int size) {
    Document doc = null;
    String docText = null;
    IndexableField field = null;
    
    INDArray input = Nd4j.zeros(new int[]{size,numCharacters,EXAMPLE_SIZE});
    INDArray labels = Nd4j.zeros(new int[]{size,numCharacters,EXAMPLE_SIZE});

    for(int i = 0; i < size; i++){
      while(doc == null || docText == null || docText.length() < EXAMPLE_SIZE+1){
        int docNum = rand.nextInt(ireader.numDocs());
        try{
        doc = ireader.document(docNum);
        }catch(IOException e){
          throw new RuntimeException(e);
        }
        field = doc.getField(textFieldName);
        if(field != null){
          docText = field.stringValue();
        }else{
          docText = null;
        }
      }        
      int startPos = rand.nextInt(docText.length() - EXAMPLE_SIZE);
      int c = 0;
      
      for(int idx = startPos; idx < startPos + EXAMPLE_SIZE; idx++, c++){
        char curChar = docText.charAt(idx);
        char nextChar = docText.charAt(idx+1);
        
        if(!charToIdxMap.containsKey(curChar) || !charToIdxMap.containsKey(nextChar)){
          continue;
        }
        
        int curCharIdx = getCharIndex(curChar);
        int nextCharIdx = getCharIndex(nextChar);
        
        input.putScalar(new int[]{i, curCharIdx, c}, 1.0);
        labels.putScalar(new int[]{i, nextCharIdx, c}, 1.0);
      }
    }
    charNum += size;
    return new DataSet(input, labels);
  }

  @Override
  public int numExamples() {
    return charNum;
  }

  @Override
  public void reset() {
    charNum = 0;
  }

  @Override
  public void setPreProcessor(DataSetPreProcessor arg0) {
    throw new UnsupportedOperationException("Not implemented");
  }

  @Override
  public int totalExamples() {
    return maxChars;
  }

  @Override
  public int totalOutcomes() {
    return this.validChars.length;
  }

  public int getCharIndex(char c){
    return charToIdxMap.get(c);
  }
  
  public char getIndexChar(int i){
    return validChars[i];
  }
  
  public char getRandomCharacter(){
    return validChars[(int) (rand.nextDouble()*validChars.length)];
  }

  public static char[] getMimicCharacterSet(){
    List<Character> validChars = new LinkedList<>();
    for(char c : CharacterIterator.getDefaultCharacterSet() ) validChars.add(c);
    char[] additionalChars = {'=', '~'};
    for( char c : additionalChars ) validChars.add(c);
    char[] out = new char[validChars.size()];
    int i=0;
    for( Character c : validChars ) out[i++] = c;
    return out;
  }
}
