# sentence-detector

This is the project that implements a character level BIO sequence tagger for sentence segmentation, as presented in an abstract title "Robust Segmentation for Clinical Text" at 
the American Medical Informatics Association 2015 Symposium.

The main entry point is a class at org/apache/ctakes/core/cleartk/eval/SentenceDetectorEvaluation.java. Other classes of interest are org/apache/ctakes/core/cleartk/ae/AnaforaSentenceXmlReader.java (for reading the data) and org/apache/ctakes/core/cleartk/ae/SentenceDetectorAnnotator.java (a UIMAFit analysis engine)
