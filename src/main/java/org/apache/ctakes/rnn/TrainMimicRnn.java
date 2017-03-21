package org.apache.ctakes.rnn;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class TrainMimicRnn {

  public static void main(String[] args) throws FileNotFoundException, IOException {
    int lstmLayerSize = 200;          //Number of units in each GravesLSTM layer
    int numEpochs = 50;             //Total number of training + sample generation epochs
    int nSamplesToGenerate = 4;         //Number of samples to generate after each training epoch
    int nCharactersToSample = 300;        //Length of each sample to generate
    String generationInitialization = null;   //Optional character initialization; a random character is used if null
    String serializedModelFilename = "mimic_rnn_model_n=200.obj";
    // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
    // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
    Random rng = new Random(12345);

    LuceneReaderCharacterIterator iter = new LuceneReaderCharacterIterator(args[0]);
    int nOut = iter.totalOutcomes();
    
    //Set up network configuration:
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
      .learningRate(0.1)
      .rmsDecay(0.95)
      .seed(12345)
      .regularization(true)
      .l2(0.001)
      .list(3)
      .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
          .updater(Updater.RMSPROP)
          .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
          .dist(new UniformDistribution(-0.08, 0.08)).build())
      .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
          .updater(Updater.RMSPROP)
          .activation("tanh").weightInit(WeightInit.DISTRIBUTION)
          .dist(new UniformDistribution(-0.08, 0.08)).build())
      .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
          .updater(Updater.RMSPROP)
          .nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
          .dist(new UniformDistribution(-0.08, 0.08)).build())
      .pretrain(false).backprop(true)
      .build();
    
    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(1));
    Layer[] layers = net.getLayers();
    int totalNumParams = 0;
    for( int i=0; i<layers.length; i++ ){
      int nParams = layers[i].numParams();
      System.out.println("Number of parameters in layer " + i + ": " + nParams);
      totalNumParams += nParams;
    }
    System.out.println("Total number of network parameters: " + totalNumParams);
    
    //Do training, and then generate and print samples from network
    for( int i=0; i<numEpochs; i++ ){
      net.fit(iter);
      
      ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File(serializedModelFilename)));
      oos.writeObject(net);
      oos.close();
      
      System.out.println("--------------------");
      System.out.println("Completed epoch " + i );
      System.out.println("Sampling characters from network given initialization \""+ (generationInitialization == null ? "" : generationInitialization) +"\"");
      String[] samples = sampleCharactersFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
      for( int j=0; j<samples.length; j++ ){
        System.out.println("----- Sample " + j + " -----");
        System.out.println(samples[j]);
        System.out.println();
      }
      
      iter.reset(); //Reset iterator for another epoch
    }
    
    System.out.println("\n\nTraining complete");
    
  }

  /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
   * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
   * Note that the initalization is used for all samples
   * @param initialization String, may be null. If null, select a random character as initialization for all samples
   * @param charactersToSample Number of characters to sample from network (excluding initialization)
   * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
   * @param iter CharacterIterator. Used for going from indexes back to characters
   */
  public static String[] sampleCharactersFromNetwork( String initialization, MultiLayerNetwork net,
      CharacterIterator_ImplBase iter, Random rng, int charactersToSample, int numSamples ){
    //Set up initialization. If no initialization: use a random character
    if( initialization == null ){
      initialization = String.valueOf(iter.getRandomCharacter(rng));
    }
    
    //Create input for initialization
    INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
    char[] init = initialization.toCharArray();
    for( int i=0; i<init.length; i++ ){
      int idx = iter.getCharIndex(init[i]);
      for( int j=0; j<numSamples; j++ ){
        initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
      }
    }
    
    StringBuilder[] sb = new StringBuilder[numSamples];
    for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);
    
    //Sample from network (and feed samples back into input) one character at a time (for all samples)
    //Sampling is done in parallel here
    net.rnnClearPreviousState();
    INDArray output = net.rnnTimeStep(initializationInput);
    output = output.tensorAlongDimension(output.size(2)-1,1,0); //Gets the last time step output
    
    for( int i=0; i<charactersToSample; i++ ){
      //Set up next input (single time step) by sampling from previous output
      INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
      //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
      for( int s=0; s<numSamples; s++ ){
        double[] outputProbDistribution = new double[iter.totalOutcomes()];
        for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
        int sampledCharacterIdx = RnnDemo.sampleFromDistribution(outputProbDistribution,new Random());
        
        nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);    //Prepare next time step input
        sb[s].append(iter.getIndexChar(sampledCharacterIdx));  //Add sampled character to StringBuilder (human readable output)
      }
      
      output = net.rnnTimeStep(nextInput);  //Do one time step of forward pass
    }
    
    String[] out = new String[numSamples];
    for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
    return out;
  }
}
