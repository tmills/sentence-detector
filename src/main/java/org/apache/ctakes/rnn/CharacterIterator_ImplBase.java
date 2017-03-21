package org.apache.ctakes.rnn;

import java.util.Map;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

public abstract class CharacterIterator_ImplBase implements DataSetIterator {

  public static final int MINI_BATCH_SIZE = 50;
  public static final int EXAMPLE_SIZE = 100;
  protected int charNum = 0;
  protected int maxChars = -1;  // the shakespeare corpus is roughly 6 million chars according to wc -c
  protected char[] validChars = null;
  protected int numCharacters = 0;
  protected Map<Character,Integer> charToIdxMap = null;

  @Override
  public abstract DataSet next(int num);

  public abstract char[] getCharacterSet();
  
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
  
  public char getRandomCharacter(Random rand){
    return validChars[(int) (rand.nextDouble()*validChars.length)];
  }
}
