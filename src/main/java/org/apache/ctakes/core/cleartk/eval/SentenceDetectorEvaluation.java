package org.apache.ctakes.core.cleartk.eval;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import org.apache.ctakes.core.cleartk.ae.AnaforaSentenceXmlReader;
import org.apache.ctakes.core.cleartk.ae.SentenceDetectorAnnotator;
import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.CASException;
import org.apache.uima.cas.Feature;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.component.JCasAnnotator_ImplBase;
import org.apache.uima.fit.component.ViewCreatorAnnotator;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.AnalysisEngineFactory;
import org.apache.uima.fit.pipeline.JCasIterator;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.TOP;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.CasCopier;
import org.cleartk.eval.AnnotationStatistics;
import org.cleartk.eval.Evaluation_ImplBase;
import org.cleartk.ml.jar.JarClassifierBuilder;
import org.cleartk.ml.liblinear.LibLinearStringOutcomeDataWriter;
import org.cleartk.util.ViewUriUtil;
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;

import com.google.common.collect.Lists;
import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

public class SentenceDetectorEvaluation extends Evaluation_ImplBase<File, AnnotationStatistics<String>> {

  static interface Options {
    @Option
    public File getAnaforaDirectory();
  }
  
  public static final String GOLD_VIEW_NAME = "GoldView";
  
  public static void main(String[] args) throws Exception {
    Options options = CliFactory.parseArguments(Options.class, args);
    
    SentenceDetectorEvaluation eval = new SentenceDetectorEvaluation(new File("target/eval"));
    
    List<File> items = getItems(options.getAnaforaDirectory());
    List<AnnotationStatistics<String>> stats = eval.crossValidation(items, 5);
    for(AnnotationStatistics<String> stat : stats){
      System.out.println("Fold: " );
      System.out.println(stat);
    }
    
    eval.train(eval.getCollectionReader(items), new File("target/eval/train_and_test"));
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
    aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(
        ViewCreatorAnnotator.class,
        ViewCreatorAnnotator.PARAM_VIEW_NAME,
        GOLD_VIEW_NAME));
    aggregateBuilder.add(UriToDocumentTextAnnotator.getDescription());
    aggregateBuilder.add(AnaforaSentenceXmlReader.getDescription(),CAS.NAME_DEFAULT_SOFA, GOLD_VIEW_NAME);
    aggregateBuilder.add(CopyFromGold.getDescription(Segment.class));
    aggregateBuilder.add(SentenceDetectorAnnotator.getDescription(directory.getAbsolutePath() + File.separator + "model.jar"));
    
    AnnotationStatistics<String> stats = new AnnotationStatistics<>();
    
    for (Iterator<JCas> casIter = new JCasIterator(collectionReader, aggregateBuilder.createAggregate()); casIter.hasNext();) {
      JCas jCas = casIter.next();
      JCas goldView = jCas.getView(GOLD_VIEW_NAME);
      JCas systemView = jCas.getView(CAS.NAME_DEFAULT_SOFA);
//      this.logger.fine("Errors in : " + ViewUriUtil.getURI(jCas).toString());
      Collection<Sentence> goldSents = JCasUtil.select(goldView, Sentence.class);
      Collection<Sentence> systemSents = JCasUtil.select(systemView, Sentence.class);
      stats.add(goldSents, systemSents);
    }
    
    return stats;
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
  
  public static class CopyFromGold extends JCasAnnotator_ImplBase {

    public static AnalysisEngineDescription getDescription(Class<?>... classes)
        throws ResourceInitializationException {
      return AnalysisEngineFactory.createEngineDescription(
          CopyFromGold.class,
          CopyFromGold.PARAM_ANNOTATION_CLASSES,
          classes);
    }

    public static final String PARAM_ANNOTATION_CLASSES = "AnnotationClasses";

    @ConfigurationParameter(name = PARAM_ANNOTATION_CLASSES, mandatory = true)
    private Class<? extends TOP>[] annotationClasses;

    @Override
    public void process(JCas jCas) throws AnalysisEngineProcessException {
      JCas goldView, systemView;
      try {
        goldView = jCas.getView(GOLD_VIEW_NAME);
        systemView = jCas.getView(CAS.NAME_DEFAULT_SOFA);
      } catch (CASException e) {
        throw new AnalysisEngineProcessException(e);
      }
      for (Class<? extends TOP> annotationClass : this.annotationClasses) {
        for (TOP annotation : Lists.newArrayList(JCasUtil.select(systemView, annotationClass))) {
          if (annotation.getClass().equals(annotationClass)) {
            annotation.removeFromIndexes();
          }
        }
      }
      CasCopier copier = new CasCopier(goldView.getCas(), systemView.getCas());
      Feature sofaFeature = jCas.getTypeSystem().getFeatureByFullName(CAS.FEATURE_FULL_NAME_SOFA);
      for (Class<? extends TOP> annotationClass : this.annotationClasses) {
        for (TOP annotation : JCasUtil.select(goldView, annotationClass)) {
          TOP copy = (TOP) copier.copyFs(annotation);
          if (copy instanceof Annotation) {
            copy.setFeatureValue(sofaFeature, systemView.getSofa());
          }
          copy.addToIndexes(systemView);
        }
      }
    }
  }
}
