package org.apache.ctakes.core.cleartk.eval;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import org.apache.ctakes.core.ae.SentenceDetector;
import org.apache.ctakes.core.cleartk.ae.AnaforaSentenceXmlReader;
import org.apache.ctakes.core.cleartk.ae.SentenceDetectorAnnotator;
import org.apache.ctakes.core.cleartk.ae.WsjSentenceReader;
import org.apache.ctakes.rnn.RnnSentenceDetector;
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

  enum FEATURE_TYPE {BASELINE, GILLICK, CHAR, SHAPE, LINE_POS, CHAR_SHAPE, CHAR_POS, CHAR_SHAPE_POS, RNN}
  enum CORPUS {MIMIC, WSJ}
  enum EVAL_TYPE { CV, DEV, TEST }
  
  static interface Options {
    @Option(shortName = "t")
    public File getTrainInputPath();

    @Option(shortName = "e", defaultToNull=true)
    public File getTestInputPath();
    
    @Option(shortName = "b")
    public boolean getBuildModel();
    
    @Option(shortName = "c", defaultValue={"MIMIC"})
    public CORPUS getCorpus();
    
    @Option(shortName = "f")
    public FEATURE_TYPE getFeatureType();    
  }
  
  public static final String GOLD_VIEW_NAME = "GoldView";
  public static Logger logger = Logger.getLogger(SentenceDetectorEvaluation.class);
  
  public static void main(String[] args) throws Exception {
    Options options = CliFactory.parseArguments(Options.class, args);
    
    SentenceDetectorEvaluation eval = new SentenceDetectorEvaluation(new File("target/eval"));
    
    List<File> trainItems = null;
    List<File> testItems = null;
    if(options.getCorpus() == CORPUS.MIMIC){
      trainItems = getAnaforaItems(options.getTrainInputPath());
      if(options.getTestInputPath() != null){
        testItems = getAnaforaItems(options.getTestInputPath());
      }
    }else if(options.getCorpus() == CORPUS.WSJ){
      trainItems = getWsjItems(options.getTrainInputPath());
      if(options.getTestInputPath() != null){
        testItems = getWsjItems(options.getTestInputPath());
      }
    }
    
    eval.evalType = options.getFeatureType();
    eval.corpus = options.getCorpus();
    
    logger.setLevel(Level.INFO);
    if(eval.evalType == FEATURE_TYPE.BASELINE) logger.setLevel(Level.INFO);

    double p,r,f;
    int tp=0, precDenom=0, recDenom=0;
    if(testItems == null){
      List<AnnotationStatistics<String>> stats = eval.crossValidation(trainItems, 5);
      for(AnnotationStatistics<String> stat : stats){
        //      System.out.println("Fold: " );
        System.out.println(stat);
        tp += stat.countCorrectOutcomes();
        precDenom += stat.countPredictedOutcomes();
        recDenom += stat.countReferenceOutcomes();
      }
    }else{
      AnnotationStatistics<String> stats = eval.trainAndTest(trainItems, testItems);
      System.out.println(stats);
      tp = stats.countCorrectOutcomes();
      precDenom = stats.countPredictedOutcomes();
      recDenom = stats.countReferenceOutcomes();
    }
    p = (double) tp / precDenom;
    r = (double) tp / recDenom;
    f = 2 * p * r / (p + r);
    System.out.println("There are " + recDenom + " gold sentences in this corpus.");
    System.out.println(String.format("Overall performance\nP\tR\tF\n%.3f\t%.3f\t%.3f", p, r, f));
    logger.setLevel(Level.WARN);
    
    if(options.getBuildModel()){
      eval.train(eval.getCollectionReader(trainItems), new File("target/eval/train_and_test"));
    }
  }

  FEATURE_TYPE evalType = FEATURE_TYPE.BASELINE;
  CORPUS corpus = CORPUS.MIMIC;
  
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
    if(evalType == FEATURE_TYPE.BASELINE) return;
    AggregateBuilder aggregateBuilder = new AggregateBuilder();
    aggregateBuilder.add(UriToDocumentTextAnnotator.getDescription());

    if(corpus == CORPUS.MIMIC){
      aggregateBuilder.add(AnaforaSentenceXmlReader.getDescription());
    }else if(corpus == CORPUS.WSJ){
      aggregateBuilder.add(WsjSentenceReader.getDescription());
    }
    
    AnalysisEngineDescription aed = null;
    
    if(evalType == FEATURE_TYPE.RNN){
      aed = RnnSentenceDetector.getDataWriter(directory, LibLinearStringOutcomeDataWriter.class);
    }else{
      aed = SentenceDetectorAnnotator.getDataWriter(directory,
      LibLinearStringOutcomeDataWriter.class);
      addParameter(aed);
    }
    
    aggregateBuilder.add(aed);
    Logger.getLogger(SentenceDetectorAnnotator.class).setLevel(Level.INFO);
    SimplePipeline.runPipeline(collectionReader, aggregateBuilder.createAggregate());
    
    
//    HideOutput hider = new HideOutput();
    JarClassifierBuilder.trainAndPackage(directory, new String[]{"-s", "2", "-c", "1.0"});
//    hider.restoreOutput();
  }

  private void addParameter(AnalysisEngineDescription aed){
    if(evalType == FEATURE_TYPE.CHAR){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.CHAR);
    }else if(evalType == FEATURE_TYPE.CHAR_SHAPE_POS){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.CHAR_SHAPE_POS);
    }else if(evalType == FEATURE_TYPE.GILLICK){      
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.GILLICK);
    }else if(evalType == FEATURE_TYPE.SHAPE){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.SHAPE);      
    }else if(evalType == FEATURE_TYPE.LINE_POS){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.LINE_POS);            
    }else if(evalType == FEATURE_TYPE.CHAR_SHAPE){
      ConfigurationParameterFactory.addConfigurationParameter(aed, 
          SentenceDetectorAnnotator.PARAM_FEAT_CONFIG, 
          SentenceDetectorAnnotator.FEAT_CONFIG.CHAR_SHAPE);            
    }else if(evalType == FEATURE_TYPE.CHAR_POS){
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
    if(corpus == CORPUS.MIMIC){
      aggregateBuilder.add(AnaforaSentenceXmlReader.getDescription(),CAS.NAME_DEFAULT_SOFA, GOLD_VIEW_NAME);
    }else if(corpus == CORPUS.WSJ){
      aggregateBuilder.add(WsjSentenceReader.getDescription(), CAS.NAME_DEFAULT_SOFA, GOLD_VIEW_NAME);
    }
    aggregateBuilder.add(CopyFromGold.getDescription(Segment.class));
    if(evalType == FEATURE_TYPE.BASELINE){
      aggregateBuilder.add(SentenceDetector.createAnnotatorDescription());
      Logger.getLogger(SentenceDetector.class).setLevel(Level.WARN);
    }else if(evalType == FEATURE_TYPE.RNN){
      aggregateBuilder.add(RnnSentenceDetector.getDescription(directory.getAbsolutePath() + File.separator + "model.jar"));
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
    
    long start = System.currentTimeMillis();
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
    long end = System.currentTimeMillis();
    logger.info("Runtime of test() for this system is: " + (end-start) + "ms");
    
    return stats;
  }

  public static List<File> getAnaforaItems(File anaforaDirectory){
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
  
  public static List<File> getWsjItems(File wsjFile){
    List<File> textFiles = new ArrayList<>();
    for(File txtFile : wsjFile.listFiles()){
      if(txtFile.getName().endsWith("raw.txt")){
        textFiles.add(txtFile);
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
