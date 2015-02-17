package org.apache.ctakes.core.cleartk.eval;

import java.io.File;
import java.util.List;

import org.apache.uima.collection.CollectionReader;
import org.cleartk.eval.AnnotationStatistics;
import org.cleartk.eval.Evaluation_ImplBase;

public class SentenceDetectorEvaluation extends Evaluation_ImplBase<File, AnnotationStatistics<String>> {

  public SentenceDetectorEvaluation(File baseDirectory) {
    super(baseDirectory);
    // TODO Auto-generated constructor stub
  }

  public static void main(String[] args) {
    // TODO Auto-generated method stub

  }

  @Override
  protected CollectionReader getCollectionReader(List<File> items)
      throws Exception {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  protected void train(CollectionReader collectionReader, File directory)
      throws Exception {
    // TODO Auto-generated method stub
    
  }

  @Override
  protected AnnotationStatistics<String> test(
      CollectionReader collectionReader, File directory) throws Exception {
    // TODO Auto-generated method stub
    return null;
  }

}
