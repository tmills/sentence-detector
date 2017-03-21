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

public class LuceneReaderCharacterIterator extends CharacterIterator_ImplBase {

  /**
   * 
   */
  private static final long serialVersionUID = 1L;
  
  public static final String textFieldName = "content";
  
  private DirectoryReader ireader = null;
  private Random rand = new Random(718);

  public LuceneReaderCharacterIterator(){
    
  }
  
  public LuceneReaderCharacterIterator(String indexDir){
    init(indexDir);
  }

  public void init(String indexDir){
    Directory dir;
    this.maxChars = MINI_BATCH_SIZE * 500;
    this.validChars = getCharacterSet();
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
  public char[] getCharacterSet(){
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
