package neuralnets;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * This neural net creates a fully connected feed-forward model. Activations are
 * propagated from left to right, and all activations are fully connected with a
 * system of weights to adjacent activations.
 * 
 * Does not utilize adaptive lambda or weight roll back.
 * Delta Too Small Termination has been removed. 
 * For bitmaps, can only process square images. 
 * 
 * Back propagation has been implemented for any number of hidden layers. 
 * 
 * Methods included: 
 * Perceptron11(PerceptronConfigurer11 config, String logFilename) throws IOException
 * void     run() throws IOException
 * void     propNetwork()
 * void     converge() throws IOException
 * void     propInputs(double[] inputs)
 * void     propActivs() 
 * double   calcError(int testCase)
 * void     backProp(int testCase) 
 * boolean  end(int iter, double currentError) throws IOException
 * double   wrapper(double n) 
 * double   wrapperPrime(double n)
 * double[] outputs()
 * void     createBitmap(int iteration, int testCase)
 * String   arrToString(double [][][] arr) 
 * String   arrToString(double[] arr) 
 * double   square(double n) 
 * 
 * @author Ellen Guo
 * @version 10 January 2020
 *
 */
public class Perceptron11 
{
   public double[][] activs; // indices: layer, row
   public double[][] thetas; // indices: layer, row; each theta corresponds to each activ
   
   public double[][] inputs; // indices: training case, i
   public double[][] targets; // indices: training case, k
   public int[] numActivs; // numActivs[n] = number of activations in layer n (layer 0 = inputs)
   public double[][][] weights; // indices: layer, left index, right index
   public int maxIterations;
   public int printOutIteration; // if this var is 10, prints out information every 10 iterations (0, 10, 20, etc.)
   public double errorThreshold;
   public double lambda;
   public String fileType; // current options: "bitmap" or "Manual" 
   public boolean squareImage; // true if the image is square, false otherwise
   
   public String logFilename;
   
   public BufferedWriter logFile;
   
   public double startTime;
   
   /**
    * Creates an instance of the object Perceptron given a set of configurations
    * 
    * @param config the PerceptronConfigurer that will dictate the configuration
    *               of this Perceptron
    * @param logFilename the name of the output file that will be created 
    * @throws IOException BufferedWriter (output log file writing) failure
    */
   public Perceptron11(PerceptronConfigurer11 config, String logFilename) throws IOException
   {
      this.inputs = config.inputs;
      this.targets = config.targets;
      this.numActivs = config.numActivs;
      this.weights = config.weights;
      this.maxIterations = config.maxIterations;
      this.printOutIteration = config.printOutIteration;
      this.errorThreshold = config.errorThreshold;
      this.lambda = config.lambda;
      this.fileType = config.fileType;
      this.squareImage = config.squareImage;
      
      this.logFilename = logFilename;
      
      this.logFile = new BufferedWriter(new FileWriter(logFilename));
      
      this.startTime = System.currentTimeMillis();
      
      //PerceptronConfigurer11.printNumPic(this.inputs[0]);
   }

   /**
    * Propagates the network and attempts to converge 
    * Creates a log file that outputs the following:
    * 1) cause of termination
    * 2) iterations
    * 3) target and calculated value and error of each case
    * 4) the final weights
    * 
    * @throws IOException BufferedWriter (output log file writing) failure
    */
   public void run() throws IOException
   {
      propNetwork();
      converge();
      logFile.close();
   }
   
   /**
    * Initializes the array of activations based on the numActivs array 
    * number of columns = length of the numActivs array (ie number of layers) 
    * number of rows = the maximum activations of the longest layer dictated by numActivs
    */
   private void propNetwork() 
   {
      int layers = numActivs.length;

      int maxNeurons = 0;
      for (int i = 0; i < numActivs.length; i++) 
      {
         if (numActivs[i] > maxNeurons)
            maxNeurons = numActivs[i];
      }

      activs = new double[layers][maxNeurons];
      thetas = new double[layers][maxNeurons];
   }
   
   /**
    * Utilizes gradient descent to train the neural network 
    * Loops through all the test cases, updating the weights with stochastic 
    * gradient descent back propagation for each case. 
    * Writes the final configuration information into a file 
    * (see run() method documentation for exactly what is written into the file) 
    * 
    * @throws IOException BufferedWriter (output log file writing) failure
    */
   public void converge() throws IOException
   {
      int iter = 0; // iter means iterations
      boolean end = iter >= maxIterations;

      propInputs(inputs[0]);
      
      propActivs();
      double prevError = calcError(0);
      
      if (fileType.equals("bitmap"))
      {
         createBitmap(-1, -1); // -1 to indicate that this is the pre-processed image
      }

      while (!end) 
      {
         for (int i = 0; i < inputs.length; i++) // loops through all the training sets
         {
            propInputs(inputs[i]);
            propActivs();

            backProp(i);
            
            // re-determine the error with the new weights by re-propagating activations
            propActivs();
            
            double newError = calcError(i);
            double errorDiff = prevError - newError;
            prevError = newError;
            
            // print out diagnostic information every printOutIteration iterations
            if (iter % printOutIteration == 0) 
            {
               double endTime = System.currentTimeMillis();
               double timeElapsed = endTime - startTime;

               System.out.print("Iteration: " + iter);
               System.out.print(" Time Elapsed since start (min): " + timeElapsed / 60000.);
               System.out.print(" New Error: " + prevError);
               System.out.print(" Change in Error: " + errorDiff);
               System.out.print(" Lambda: " + lambda);
               System.out.print(" Case: " + i);
               System.out.println(" Output: " + arrToString(activs[activs.length - 1]) 
                  + " Target: " + arrToString(targets[i]));
               
               // create diagnostic bitmap if the input is a bitmap
               if (fileType.equals("bitmap"))
               {
                  for (int testCase = 0; testCase < inputs.length; testCase++)
                  {
                     //createBitmap(iter, testCase);
                  }
               }
            } // if (iter % printOutIteration == 0)
            
            iter++;
            
         } //for (int i = 0; i < inputs.length; i++)
         
         //calculates the maximum error across all test cases
         double maxError = 0.0;
         
         for (int i = 0; i < inputs.length; i++)
         {
            propInputs(inputs[i]);
            propActivs();
            
            double error = calcError(i);
            
            if (error > maxError) 
            {
               maxError = error;
            }
         }
         
         end = end(iter, maxError);
         
      } // while (!end) 

      // after termination, writes the output log file 
      
      System.out.println("Termination Reached. Writing output file now.");
      
      logFile.write("Iterations: " + iter + "\n\n");
      
      // writes the target and calculated values and errors for each test case
      for (int testCase = 0; testCase < targets.length; testCase++) 
      {
         logFile.write("Case " + testCase + ": Target / Calculated \n");
         propInputs(inputs[testCase]);
         propActivs();
         
         for (int output = 0; output < targets[0].length; output++)
         {
            logFile.write(targets[testCase][output] + " ");
            logFile.write(activs[activs.length - 1][output] + "\n");
         }
         
         logFile.write("Case " + testCase + " total error: " + calcError(testCase) + "\n\n");
         
      } // for (int testCase = 0; testCase < targets.length; testCase++) 
      
      //logFile.write("Final Weights: \n" + arrToString(weights));
      
   } // public void converge() throws IOException

   /**
    * Sets the first layer of activations (input layer) to given array of inputs
    * 
    * @param inputs the array to set as inputs
    */
   private void propInputs(double[] inputs) 
   {
      for (int i = 0; i < inputs.length; i++) 
      {
         activs[0][i] = inputs[i];
      }
   }
   
   /**
    * Propagates all the activations within the neural net: Computes each
    * activation by summing the product of each activation in the preceding layer
    * with the corresponding weight, then compressing the value with a wrapper
    * function. Also stores an array of thetas; for every activation, there is a 
    * corresponding theta that is the activation value pre-wrapper. 
    * 
    * How the indices and loops work: loop through each layer of activations
    * ("layer") within each layer, loop through every activation in that layer
    * ("right", ie. the destination) for each of these activations, compute its
    * value by looping through each activation in the previous layer ("left", ie
    * source) and multiplying with the corresponding weight [layer - 1][left][right]
    */
   private void propActivs() 
   {
      for (int layer = 1; layer < numActivs.length; layer++) 
      {
         for (int right = 0; right < numActivs[layer]; right++) 
         {
            double theta = 0.0;

            for (int left = 0; left < numActivs[layer - 1]; left++) 
            {
               theta += activs[layer - 1][left] * weights[layer - 1][left][right];
            }
            
            thetas[layer][right] = theta;
            activs[layer][right] = wrapper(theta);
            
         } // for (int right = 0; right < numActivs[layer]; right++)
      } // for (int layer = 1; layer < numActivs.length; layer++) 
   } // private void propActivs() 
   
   /**
    * Calculates the error of the current network configuration 
    * with regards to a given training set 
    * 
    * @param testCase the index of the target value array to compare against 
    * @return the error, as defined as half the sum of the squares of the 
    *         differences between each real and calculated value
    */
   private double calcError(int testCase) 
   {
      double error = 0.0;
      
      double[] targetOutputs = targets[testCase];
      double[] outputActivs = activs[activs.length - 1];
      
      for (int outputIndex = 0; outputIndex < numActivs[numActivs.length - 1]; outputIndex++)
      {
         error += square(targetOutputs[outputIndex] - outputActivs[outputIndex]);
      }
      return 0.5 * error;
   } //private double calcError(int testCase) 
   
   /**
    * Updates the weights using stochastic gradient descent and back propagation
    * Can be used with any number of activations in the input, hidden, and output layers
    * Can be used with any number of hidden layers
    * 
    * @param testCase the index in the "inputs" array that holds the array of target values Ti
    *            that the back prop is based on 
    */
   private void backProp(int testCase) 
   {
      double[][] bigOmega = new double[activs.length][activs[0].length]; // indices are [layer][index]
      
      // update last layer of weights first
      int jlayer = numActivs.length - 2; // -2 represents the second last layer
      
      for (int j = 0; j < numActivs[jlayer]; j++)
      {
         double bigOmegaj = 0.0; 
         
         for (int i = 0; i < numActivs[jlayer + 1]; i++)
         {
            double thetai = thetas[thetas.length - 1][i];
            double Fi = wrapper(thetai);
            double littleOmegai = (targets[testCase][i]) - Fi;
            double psii = littleOmegai * wrapperPrime(thetai);
            double hj = activs[activs.length - 2][j];
            
            bigOmegaj += psii * weights[jlayer][j][i];
            
            weights[jlayer][j][i] += lambda * hj * psii;
         }
         
         bigOmega[jlayer][j] = bigOmegaj;
         
      } // for (int j = 0; j < numActivs[jlayer]; j++)
      
      int layers = weights.length;
      
      // calculate and update weights for all the other layers
      for (int layer = layers - 2; layer >= 0; layer--) 
      {
         for (int k = 0; k < numActivs[layer]; k++)
         {
            double bigOmegak = 0.0;
            
            for (int j = 0; j < numActivs[layer + 1]; j++)
            {
               double bigPsij = bigOmega[layer + 1][j] * wrapperPrime(thetas[layer + 1][j]);
               double ak = activs[layer][k];

               bigOmegak += bigPsij * weights[layer][k][j];
               weights[layer][k][j] += lambda * ak * bigPsij;
            }
            
            bigOmega[layer][k] = bigOmegak;
            
         } // for (int k = 0; k < numActivs[layer]; k++)
      } // for (int layer = layers - 2; layer >= 0; layer--)
   } // private void backProp(int testCase) 

   /**
    * Determines if the given conditions should cause termination.
    * If so, writes the reason for termination into the output log file
    * 
    * Timeout Termination: when the program has been through too many iterations
    * Convergence Termination: when the maximum error across all training sets 
    *    has fallen beneath a given threshold
    * Lambda = 0 Termination: when lambda = 0, the change in weights becomes negligible as well 
    * 
    * Does NOT include Delta Too Small Termination 
    * 
    * @param iter the number of iterations the network has been through already 
    * @param currentError the maximum error across all training sets
    * @throws IOException BufferedWriter (output log file writing) failure
    * @return true if any of the four termination conditions are reached
    *     false otherwise
    */
   private boolean end(int iter, double currentError) throws IOException
   {
      boolean end = false;

      if (iter >= maxIterations) 
      {
         logFile.write("Cause of Termination: Timeout Error \n\n");
         end = true;
      } 
      else if (currentError < errorThreshold) 
      {
         logFile.write("Cause of Termination: Error Threshold (" + errorThreshold + ") "
               + "reached for all cases \n\n");
         end = true;
      }
      else if (lambda == 0) 
      {
         logFile.write("Cause of Termination: Lambda = 0\n\n");
         end = true;
      }
      return end;
   } // private boolean end(int iter, double currentError) throws IOException

   /**
    * A wrapper function that compresses a given value to a desired range
    * Wraps around every activation 
    * 
    * currently: uses a sigmoid, which goes from [R] --> (0,1)
    * can be changed to other functions, such as the hyperbolic tangent function
    * 
    * @param n the given value to compress
    * @return the compressed value
    */
   public double wrapper(double n) 
   {
      return 1.0 / (1.0 + Math.exp(-n));
   }
   
   /**
    * The derivative of the wrapper function
    * 
    * currently: the derivative of the sigmoid function
    * 
    * @param n the input for the derivative
    * @return output of the derivative
    */
   private double wrapperPrime(double n)
   {
      double wrapped = wrapper(n);
      return wrapped * (1.0 - wrapped);
   }
   
   /**
    * Gets the activation values in the output layer
    * 
    * @return a 1D array of outputs 
    */
   public double[] outputs()
   {
      double[] outputs = new double[numActivs[numActivs.length - 1]];
      
      for (int i = 0; i < outputs.length; i++)
      {
         outputs[i] = activs[activs.length - 1][i];
      }
      
      return outputs;
   }
   
   /**
    * Sets up the parameters for a call to DibDump that creates a true color bitmap
    * file on the disc using the current outputs. Converts the output doubles into 
    * pel values (essentially the reverse of the pel --> [0,1] scaling). 
    * 
    * @param iteration the number of iterations the network has been through already
    * @param testCase the training set that this iteration had been run with
    */
   public void createBitmap(int iteration, int testCase)
   {
      DibDump1 dib = new DibDump1();
      
      int count = 0;
      
      int height = 0;
      int width = 0;
      
      if (squareImage)
      {
         height = (int)Math.sqrt(numActivs[0]);
         width = (int)Math.sqrt(numActivs[0]);
      }
      else
      {
         throw new RuntimeException("Not a square image. Don't know how to process.");
      }
      
      int[][] outputs = new int[height][width];
      
      for (int a = 0; a < outputs[0].length; a++)
      {
         for (int b = 0; b < outputs[0].length; b++)
         {
            outputs[a][b] = (int) PerceptronConfigurer11.reverseScale(activs[activs.length - 1][count]);
            count++;
         }
      }
      
      String filename = logFilename.substring(0, logFilename.indexOf(".")) + "_" + iteration + "_" + maxIterations + 
            "testCase" + testCase + ".bmp";
      
      dib.writeOut(outputs, filename);
      
   } // public void createBitmap(int iteration, int testCase)

   /**
    * Creates a string representation of a 3D array 
    * (overloads the other arrToString method)
    * 
    * @precondition the arr must be the same size as the weights array 
    * 
    * @param arr the given array from which the method creates the string
    * @return the string representation of the array 
    */
   public String arrToString(double [][][] arr) 
   {
      String s = "";

      for (int layer = 0; layer < numActivs.length - 1; layer++) 
      {
         for (int left = 0; left < numActivs[layer]; left++) 
         {
            for (int right = 0; right < numActivs[layer + 1]; right++) 
            {
               s += "w[" + layer + "][" + left + "][" + right + "] = " + arr[layer][left][right] + "\n";
            }
         }
      }
      return s;
   }
   
   /**
    * Creates a string representation of a 1D array 
    * (overloads the other arrToString method)
    * 
    * @param arr the given array from which the method creates the string
    * @return the string representation of the array 
    */
   public String arrToString(double[] arr) 
   {
      String s =  "[";
      
      for (int i = 0; i < arr.length - 1; i++) 
      {
         s += arr[i] + ", ";
      }
      s += arr[arr.length - 1] + "]";
      
      return s;
   }

   /**
    * Squares a given number by multiplying it with itself
    * Reduces errors associated with floating point arithmetic that may occur when using
    * Math.square()
    * 
    * @param n the given number to square
    * @return the square of the given number 
    */
   private double square(double n) 
   {
      return n * n;
   }
   
} // public class Perceptron11 