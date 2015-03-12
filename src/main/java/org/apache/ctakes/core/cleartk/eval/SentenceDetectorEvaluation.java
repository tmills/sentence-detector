package org.apache.ctakes.core.cleartk.eval;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.apache.ctakes.core.ae.DocumentIdPrinterAnalysisEngine;
import org.apache.ctakes.core.ae.SentenceDetector;
import org.apache.ctakes.core.cleartk.ae.AnaforaSentenceXmlReader;
import org.apache.ctakes.core.cleartk.ae.SentenceDetectorAnnotator;
import org.apache.ctakes.typesystem.type.textspan.Segment;
import org.apache.ctakes.typesystem.type.textspan.Sentence;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
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
import org.apache.uima.fit.factory.ConfigurationParameterFactory;
import org.apache.uima.fit.pipeline.JCasIterator;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.testing.util.HideOutput;
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
import org.cleartk.util.ae.UriToDocumentTextAnnotator;
import org.cleartk.util.cr.UriCollectionReader;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

public class SentenceDetectorEvaluation extends Evaluation_ImplBase<File, AnnotationStatistics<String>> {

  enum EVAL_TYPE {BASELINE, GILLICK, CHAR, SHAPE, LINE_POS, CHAR_SHAPE, CHAR_POS, CHAR_SHAPE_POS}
  
  static interface Options {
    @Option
    public File getAnaforaDirectory();
    
    @Option(shortName = "-b")
    public boolean getBuildModel();
    
    @Option(shortName = "-c")
    public EVAL_TYPE getEvalType();
  }
  
  public static final String GOLD_VIEW_NAME = "GoldView";
  public static Logger logger = Logger.getLogger(SentenceDetectorEvaluation.class);
  
  public static void main(String[] args) throws Exception {
    Options options = CliFactory.parseArguments(Options.class, args);
    
    SentenceDetectorEvaluation eval = new SentenceDetectorEvaluation(new File("target/eval"));
    
    List<File> items = getItems(options.getAnaforaDirectory());
    eval.evalType = options.getEvalType();
    logger.setLevel(Level.DEBUG);
    if(eval.evalType == EVAL_TYPE.BASELINE) logger.setLevel(Level.WARN);

    List<AnnotationStatistics<String>> stats = eval.crossValidation(items, 5);
    double p,r,f;
    int tp=0, precDenom=0, recDenom=0;
    for(AnnotationStatistics<String> stat : stats){
      //      System.out.println("Fold: " );
      System.out.println(stat);
      tp += stat.countCorrectOutcomes();
      precDenom += stat.countPredictedOutcomes();
      recDenom += stat.countReferenceOutcomes();
    }
    p = (double) tp / precDenom;
    r = (double) tp / recDenom;
    f = 2 * p * r / (p + r);
    System.out.println("There are " + recDenom + " gold sentences in this corpus.");
    System.out.println(String.format("Overall performance\nP\tR\tF\n%.3f\t%.3f\t%.3f", p, r, f));
    logger.setLevel(Level.WARN);
    
    if(options.getBuildModel()){
      eval.train(eval.getCollectionReader(items), new File("target/eval/train_and_test"));
    }
  }

  EVAL_TYPE evalType = EVAL_TYPE.BASELINE;
  
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
    if(evalType == EVAL_TYPE.BASELINE) return;
    AggregateBuilder aggregateBuilder = new AggregateBuilder();
    aggregateBuilder.add(UriToDocumentTextAnnotator.getDescription());
    aggregateBuilder.add(AnaforaSentenceXmlReader.getDescription());
    AnalysisEngineDescription aed = SentenceDetectorAnnotator.getDataWriter(directory,
        LibLinearStringOutcomeDataWriter.class);
    addParameter(aed);

    aggregateBuilder.add(aed);
    Logger.getLogger(SentenceDetectorAnnotator.class).setLevel(Level.INFO);
    SimplePipeline.runPipeline(collectionReader, aggregateBuilder.createAggregate());
    
    
    HideOutput hider = new HideOutput();
    JarClassifierBuilder.trainAndPackage(directory, new String[]{"-c", "0.1"});
    hider.restoreOutput();
  }

  private void addParameter(AnalysisEngineDescription aed){
    if(evalType == EVAL_TYPE.CHAR){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.CHAR);
    }else if(evalType == EVAL_TYPE.CHAR_SHAPE_POS){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.CHAR_SHAPE_POS);
    }else if(evalType == EVAL_TYPE.GILLICK){      
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.GILLICK);
    }else if(evalType == EVAL_TYPE.SHAPE){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.SHAPE);      
    }else if(evalType == EVAL_TYPE.LINE_POS){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.LINE_POS);            
    }else if(evalType == EVAL_TYPE.CHAR_SHAPE){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.CHAR_SHAPE);            
    }else if(evalType == EVAL_TYPE.CHAR_POS){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.CHAR_POS);            
    }
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
    if(evalType == EVAL_TYPE.BASELINE){
      aggregateBuilder.add(SentenceDetector.createAnnotatorDescription());
      Logger.getLogger(SentenceDetector.class).setLevel(Level.WARN);
    }else{
      AnalysisEngineDescription aed =  SentenceDetectorAnnotator.getDescription(directory.getAbsolutePath() + File.separator + "model.jar");
      addParameter(aed);
      aggregateBuilder.add(aed);
      Logger.getLogger(SentenceDetectorAnnotator.class).setLevel(Level.INFO);
    }
    aggregateBuilder.add(AnalysisEngineFactory.createEngineDescription(SentenceBoundaryAdjuster.class));
    
    AnnotationStatistics<String> stats = new AnnotationStatistics<>();
    Ordering<Annotation> bySpans = Ordering.<Integer> natural().lexicographical().onResultOf(
        new Function<Annotation, List<Integer>>() {
          @Override
          public List<Integer> apply(Annotation annotation) {
            return Arrays.asList(annotation.getBegin(), annotation.getEnd());
          }
        });
    
    for (Iterator<JCas> casIter = new JCasIterator(collectionReader, aggregateBuilder.createAggregate()); casIter.hasNext();) {
      JCas jCas = casIter.next();
      JCas goldView = jCas.getView(GOLD_VIEW_NAME);
      JCas systemView = jCas.getView(CAS.NAME_DEFAULT_SOFA);
//      this.logger.fine("Errors in : " + ViewUriUtil.getURI(jCas).toString());
      Collection<Sentence> goldSents = JCasUtil.select(goldView, Sentence.class);
      Collection<Sentence> systemSents = JCasUtil.select(systemView, Sentence.class);
      stats.add(goldSents, systemSents);
      
      Set<Annotation> goldOnly = new TreeSet<Annotation>(bySpans);
      goldOnly.addAll(goldSents);
      for(Annotation systemSent : systemSents){
        goldOnly.remove(systemSent);
      }

      Set<Annotation> systemOnly = new TreeSet<Annotation>(bySpans);
      systemOnly.addAll(systemSents);
      // this is necessary because of weird removal logic that only uses bySpans for equality if
      // the argument collection is smaller than the calling collection.
      for(Annotation goldSent : goldSents){
        systemOnly.remove(goldSent);
      }

      String text = jCas.getDocumentText().replace('\n', 'ß');
      String label = "DROPPED";
      for(Annotation annotation : goldOnly){
        int begin = annotation.getBegin();
        int end = annotation.getEnd();
        int windowBegin = Math.max(0, begin - 50);
        int windowEnd = Math.min(text.length(), end + 50);
        logger.debug(String.format(
            "%s  ...%s[!%s!:%d-%d]%s...",
            label,
            text.substring(windowBegin, begin),
            text.substring(begin, end),
            begin,
            end,
            text.substring(end, windowEnd)));
      }
      label = "ADDED";
      for(Annotation annotation : systemOnly){
        int begin = annotation.getBegin();
        int end = annotation.getEnd();
        int windowBegin = Math.max(0, begin - 50);
        int windowEnd = Math.min(text.length(), end + 50);
        logger.debug(String.format(
            "%s  ...%s[!%s!:%d-%d]%s...",
            label,
            text.substring(windowBegin, begin),
            text.substring(begin, end),
            begin,
            end,
            text.substring(end, windowEnd)));

      }
    }
    
    return stats;
  }

  public static List<File> getItems(File anaforaDirectory){
    List<File> textFiles = new ArrayList<>();
    for(File subDir : anaforaDirectory.listFiles()){
      File[] anaFiles = subDir.listFiles();
      for(File anaFile : anaFiles){
        if(anaFile.getName().endsWith("completed.xml")){
          textFiles.add(new File(subDir, subDir.getName()));
          break;
        }
      }
    }
    return textFiles;
  }
  
  public static class SentenceBoundaryAdjuster extends JCasAnnotator_ImplBase{ 
    @Override
    public void process(JCas jcas) throws AnalysisEngineProcessException {
      String docText = jcas.getDocumentText();
      for(Sentence sent : JCasUtil.select(jcas, Sentence.class)){
        switch(docText.charAt(sent.getEnd()-1)){
        case '.':
        case '?':
        case '!':
          sent.setEnd(sent.getEnd()-1);
        }
      }
    }
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
