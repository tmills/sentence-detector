package org.apache.ctakes.core.cleartk.eval;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.ctakes.core.ae.DocumentIdPrinterAnalysisEngine;
import org.apache.ctakes.core.cleartk.ae.AnaforaSentenceXmlReader;
import org.apache.ctakes.core.cleartk.ae.SentenceDetectorAnnotator;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import static org.apache.uima.fit.factory.AnalysisEngineFactory.*;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.cleartk.eval.AnnotationStatistics;
import org.cleartk.eval.Evaluation_ImplBase;
import org.cleartk.ml.jar.JarClassifierBuilder;
import org.cleartk.ml.liblinear.LibLinearStringOutcomeDataWriter;
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;

import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

public class SentenceDetectorEvaluation extends Evaluation_ImplBase<File, AnnotationStatistics<String>> {

  static interface Options {
    @Option
    public File getAnaforaDirectory();
  }
  
  public static void main(String[] args) throws Exception {
    Options options = CliFactory.parseArguments(Options.class, args);
    
    SentenceDetectorEvaluation eval = new SentenceDetectorEvaluation(new File("target/eval"));
    
    List<File> items = getItems(options.getAnaforaDirectory());
    List<AnnotationStatistics<String>> stats = eval.crossValidation(items, 5);
    for(AnnotationStatistics<String> stat : stats){
      System.out.println("Fold: " );
      System.out.println(stat);
    }
  }

  public SentenceDetectorEvaluation(File baseDirectory) {
    super(baseDirectory);
  }

  @Override
  protected CollectionReader getCollectionReader(List<File> items)
      throws Exception {
    return UriCollectionReader.getCollectionReaderFromFiles(items);
  }

  @Override
  protected void train(CollectionReader collectionReader, File directory)
      throws Exception {
    AggregateBuilder aggregateBuilder = new AggregateBuilder();
    aggregateBuilder.add(UriToDocumentTextAnnotator.getDescription());
    aggregateBuilder.add(AnaforaSentenceXmlReader.getDescription());
    aggregateBuilder.add(SentenceDetectorAnnotator.getDataWriter(directory,
        LibLinearStringOutcomeDataWriter.class));
    
    SimplePipeline.runPipeline(collectionReader, aggregateBuilder.createAggregate());
    
    JarClassifierBuilder.trainAndPackage(directory, new String[]{"-c", "1"});
  }

  @Override
  protected AnnotationStatistics<String> test(
      CollectionReader collectionReader, File directory) throws Exception {
    AggregateBuilder aggregateBuilder = new AggregateBuilder();
    aggregateBuilder.add(UriToDocumentTextAnnotator.getDescription());
    aggregateBuilder.add(AnaforaSentenceXmlReader.getDescription());
    
    return null;
  }

  private static List<File> getItems(File anaforaDirectory){
    List<File> textFiles = new ArrayList<>();
    for(File subDir : anaforaDirectory.listFiles()){
      File[] anaFiles = subDir.listFiles();
      if(anaFiles.length > 1){
        textFiles.add(new File(subDir, subDir.getName()));
      }
    }
    return textFiles;
  }
}
