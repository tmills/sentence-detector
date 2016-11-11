# sentence-detector

This is the project that implements a character level BIO sequence tagger for sentence segmentation, as presented in an abstract title "Robust Segmentation for Clinical Text" at 
the American Medical Informatics Association 2015 Symposium.

The main entry point is a class at org/apache/ctakes/core/cleartk/eval/SentenceDetectorEvaluation.java. Other classes of interest are org/apache/ctakes/core/cleartk/ae/AnaforaSentenceXmlReader.java (for reading the data) and org/apache/ctakes/core/cleartk/ae/SentenceDetectorAnnotator.java (a UIMAFit analysis engine).

To completely reproduce these results, you would need the following:

1) The source data from MIMIC V2 (http://physionet.org/mimic2/)
2) The anafora xml (contained in a different repo: https://github.com/tmills/clinical_sentence_annotations)
3) Code to extract the MIMIC V2 data into a set of directories with the same naming conventions as the anafora xml files.
4) A script to merge the raw text data from your extracted MIMIC data with the xml as the anafora reader expects (I may write something to do this). The basic format is that each directory needs to have the source text file (with no extension) and each annotation adds several extensions that represent annotation metadata (<Root Filename>.<Annotation Schema>.<Annotator>.<Status>.xml)

To run the evaluation one needs to run the SentenceDetectorEvaluation class above. I do this within eclipse with the m2e plugin so that maven handles all the libraries and sets up the classpath. The command line also requires two arguments:

--anaforaDirectory <path to folder with anafora files>
--evalType <BASELINE|GILLICK|CHAR|SHAPE|LINE_POS|CHAR_SHAPE|CHAR_POS|CHAR_SHAPE_POS|RNN>

There is an optional flag you can pass:
-b

If this flag is set the code will run a 5-fold cross validation first but after evaluating it will train a model on all of the data.

