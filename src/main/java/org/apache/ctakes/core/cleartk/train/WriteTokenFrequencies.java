package org.apache.ctakes.core.cleartk.train;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.ctakes.core.cleartk.ae.AnaforaSentenceXmlReader;
import org.apache.ctakes.core.cleartk.eval.SentenceDetectorEvaluation;
import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.ctakes.utils.struct.CounterMap;
import org.apache.uima.UIMAException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.pipeline.JCasIterator;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;

import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

public class WriteTokenFrequencies {
  static interface Options {
    @Option
    public File getAnaforaDirectory();
    
    @Option
    public String getOutputFile();
  }
  
  public static void main(String[] args) throws UIMAException, IOException {
    Options options = CliFactory.parseArguments(Options.class, args);
    List<File> items = SentenceDetectorEvaluation.getItems(options.getAnaforaDirectory());

    CollectionReader collectionReader = UriCollectionReader.getCollectionReaderFromFiles(items);
    AggregateBuilder aggregateBuilder = new AggregateBuilder();
    aggregateBuilder.add(UriToDocumentTextAnnotator.getDescription());
    aggregateBuilder.add(AnaforaSentenceXmlReader.getDescription());

    JCasIterator casIter = new JCasIterator(collectionReader, aggregateBuilder.createAggregate());
    CounterMap<String> tokenCounts = new CounterMap<>();
    
    while(casIter.hasNext()){
      JCas jcas = casIter.next();
      for(Segment seg : JCasUtil.select(jcas, Segment.class)){
        for(Sentence sent : JCasUtil.selectCovered(Sentence.class, seg)){
          String[] tokens = sent.getCoveredText().split("\\s+");
          for(String token : tokens){
            if(token.matches("\\p{Alpha}+\\p{Punct}")){
              token = token.substring(0, token.length()-1);
            }
            if(token.length() == 0) continue;
            tokenCounts.add(token);
          }
        }
      }
    }
    
    PrintWriter out = new PrintWriter(options.getOutputFile());
    for(String key : tokenCounts.keySet()){
      out.print(String.format("%s : %d\n", key, tokenCounts.get(key)));
    }
    out.close();
  }
}
