package neuralnets;
import java.util.Scanner;
import java.io.File; 
import java.io.FileNotFoundException; 
import java.io.IOException;

/**
 * Given a file, will parse it to extract configuration for a Perceptron object
 * See README for file specifications
 * 
 * NOTE: Scanner.nextDouble() wasn't recognizing 0.0 for me, so I had to resort to 
 * reading 0 with Scanner.nextInt() and then casting to a double
 * 
 * Methods included:
 * PerceptronConfigurer11(File file) throws RuntimeException, FileNotFoundException, IOException
 * int[][] handProcess(int[][] pels)
 * void printNumPic(double[] arr)
 * double[][][] propWeightsRand(double[][][] w8s, double min, double max) 
 * double random(double min, double max)
 * double scale(double n)
 * double reverseScale(double n)
 * 
 * @author Ellen Guo
 * @version 10 January 2020
 *
 */
public class PerceptronConfigurer11 
{
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
   
   private static double numBits; // number of bits for each pel
   
   /**
    * Creates an instance of the PerceptronConfigurer11 class given a file 
    * Reads through the file, utilizing the headings, and instantiates 
    * instance variables based on the file configurations
    * 
    * @param file the file to read
    * @throws RuntimeException if file cannot be understood in the context of creating a Perceptron
    *        ie. does not meet specifications
    * @throws FileNotFoundException 
    */
   public PerceptronConfigurer11(File file) throws RuntimeException, FileNotFoundException, IOException
   {
      Scanner scan = new Scanner(file);
      
      int numTestCases = 0;
      
      while (scan.hasNext())
      {
         String line = scan.nextLine();
         
         if (!line.isEmpty())
         {
            int colonIndex = line.indexOf(":");
            String label = "";
            
            if (colonIndex > -1) 
            {
               label = line.substring(0, colonIndex);
            
               if (label.equals("numTestCases"))
               {
                  numTestCases = Integer.parseInt(line.substring(colonIndex + 2));
               }
               else if (label.equals("numHiddenLayers"))
               {
                  int numHiddenLayers = Integer.parseInt(line.substring(colonIndex + 2));
                  numActivs = new int[numHiddenLayers + 2]; // 2 represents the addition of 1 input layer and 1 output layer
                  
                  for (int hiddenLayer = 0; hiddenLayer < numHiddenLayers; hiddenLayer++)
                  {
                     numActivs[hiddenLayer + 1] = scan.nextInt();
                  }
               }
               else if (label.equals("bitmap"))
               {
                  String bitmap = line.substring(colonIndex + 2);
                  
                  if (bitmap.equals("yes"))
                  {
                     fileType = "bitmap";
                     
                     DibDump1 dib = new DibDump1();
                     
                     // read in if the image is color or grayscale
                     String color = scan.nextLine();
                     if (color.equals("grayscale"))
                     {
                        numBits = 255.0; // 2^8 - 1
                     }
                     else if (color.equals("RGB"))
                     {
                        numBits = 16777215.0; // 2^24 - 1 
                     }
                     
                     // read in if the image is square or not ALL FILES MUST BE SQUARE FOR "SQUARE"
                     String square = scan.nextLine();
                     if (square.equals("square"))
                     {
                        squareImage = true;
                     }
                     else if (square.equals("not square"))
                     {
                        squareImage = false;
                     }
                     else
                     {
                        throw new RuntimeException("Invalid square image condition: \"" + square + "\"");
                     }
                     
                     boolean compression = false; //true if want to output another image; false if want to output numbers
                     
                     // read in what type of output we want
                     String outputType = scan.nextLine();
                     if (outputType.equals("image")) // for image compression
                     {
                        compression = true;
                     }
                     else if (outputType.equals("number")) // for image classification
                     {
                        compression = false;
                     }
                     else
                     {
                        throw new RuntimeException("Invalid output type: \"" + outputType + "\"");
                     }
                     
//                     String[] images = new String[numTestCases];
//                     
//                     for (int i = 0; i < numTestCases; i++)
//                     {
//                        images[i] = scan.nextLine();
//                     }
                     
                     String process = scan.nextLine();
                     boolean handProcessing = false;
                     
                     if (process.equals("hand"))
                     {
                        handProcessing = true;
                     }
                     else if (! process.equals("letter"))
                     {
                        throw new RuntimeException("Invalid image processing procedure: \"" + process + "\"");
                     }
                     
                     int numPelsInput = scan.nextInt();
                     
                     inputs = new double[numTestCases][numPelsInput];
                     
                     scan.nextLine();
                     
                  // loop through the list of images that follows and retrieve the pels from these images
                     for (int images = 0; images < numTestCases; images++)
                     {
                        String image = scan.nextLine();
                        System.out.println(image);
                        
                        String[] arguments = {image};
                        dib.main(arguments);
                  
                        int[][] dibInputs = dib.imageArray;
                        
                        System.out.println();
                        
                        if (handProcessing)
                        {
                           dibInputs = this.handProcess(dibInputs);
                        }
                        
                        dib.writeOut(dibInputs, "DEBUG" + images + ".bmp");
                        
                        int rows = dibInputs.length;
                        int cols = dibInputs[0].length;
                        
                        int numPels = rows * cols;
                        
//                        System.out.println("DEBUG testcases: " + numTestCases);
                        
                        if (numPels != inputs[0].length)
                        {
                           throw new RuntimeException("numPels doesn't match size of image" + numPels + " " + inputs[0].length);
                        }
                        
                        numActivs[0] = numPels;
                        
                        // Instantiate targets array and add output layer information to numActivs array
                        if (compression)
                        {
                           targets = new double[numTestCases][numPels];
                           numActivs[numActivs.length - 1] = numPels;
                        }
                        else
                        {
                           // magic #s here because assume 1 output if doing image classification 
                           targets = new double[numTestCases][1];
                           numActivs[numActivs.length - 1] = 1;
                        }
                        
                        int count = 0;
                        
                        // scale the extracted pel values and set them as the inputs and/or targets
                        for (int i = 0; i < rows; i++) 
                        {
                           for (int j = 0; j < cols; j++) 
                           {
                              double pel = (double)(dibInputs[i][j]);
                              
                              //System.out.println("DEBUG pel: " + pel);
                              
                              double num = scale(pel);
                              
                              //System.out.println("DEBUG scaled: " + num);
                              
                              if (compression) 
                              {
                                 targets[images][count] = num;
                              }
                              
                              inputs[images][count] = num;
                              
                              count++;
                              
                           } // for (int j = 0; j < cols; j++)
                        } // for (int i = 0; i < rows; i++) 

                        //this.printNumPic(inputs[0]);
                        
                     } // for (int images = 0; images < numTestCases; images++)

                     if (!compression)
                     {
                        for (int i = 0; i < numTestCases; i++)
                        {
                           targets[i][0] = scan.nextDouble(); // index 0 because assume 1 output 
                        }
                     }
                     
                  } // if (bitmap.equals("yes"))
                  
                  else if (bitmap.equals("no"))
                  {
                     fileType = "Manual";
                     
                     if (numTestCases <= 0)
                     {
                        throw new RuntimeException("numTestCases must preceded numInputs and numOutputs in file");
                     }
                     else
                     {
                        int numInputs = scan.nextInt();
                        inputs = new double[numTestCases][numInputs];
                        numActivs[0] = numInputs;
                        
                        // propagates the inputs array
                        for (int testCase = 0; testCase < numTestCases; testCase++)
                        {
                           for (int input = 0; input < numActivs[0]; input++)
                           {
                              inputs[testCase][input] = (double)(scan.nextInt());
                           }
                        }
                        
                        int numOutputs = scan.nextInt();
                        targets = new double[numTestCases][numOutputs];
                        numActivs[numActivs.length - 1] = numOutputs;
                        
                        // propagates the targets array 
                        for (int testCase = 0; testCase < numTestCases; testCase++)
                        {
                           for (int output = 0; output < numActivs[numActivs.length - 1]; output++)
                           {
                              double target = (double)(scan.nextInt());
                              targets[testCase][output] = target;
                           }
                        }
                     } // else clause where numTestCases > 0
                  } // else if (bitmap.equals("no")
                  else 
                  {
                     throw new RuntimeException("Invalid bitmap condition: \"" + bitmap + "\"");
                  }
               }
               else if (label.equals("weights")) // gets weights
               {
                  if (numActivs == null)
                  {
                     throw new RuntimeException("numHiddenLayers must precede weights in file");
                  }
                  else if (inputs == null)
                  {
                     throw new RuntimeException("inputs must precede weights in file");
                  }
                  else if (targets == null)
                  {
                     throw new RuntimeException("outputs must precede weights in file");
                  }
                  else 
                  {
                     String weightPropStyle = line.substring(colonIndex + 2);
                     
                     // determine the maximum number of activations in a layer (ie. how many are in the longest layer)
                     int maxActivs = 0;
                     for (int i = 0; i < numActivs.length; i++)
                     {
                        maxActivs = Math.max(maxActivs, numActivs[i]);
                     }
                     
                     System.out.println("DEBUG maxActivs: " + maxActivs);
                     System.out.println("DEBUG num layers: " + (numActivs.length - 1));
                     
                     weights = new double[numActivs.length - 1][maxActivs][maxActivs];
                     
                     if (weightPropStyle.equals("Random")) // creates a set of random weights
                     {
                        weights = propWeightsRand(weights, scan.nextDouble(), scan.nextDouble());
                     }
                     else if (weightPropStyle.equals("Manual")) // reads user entered weights
                     {
                        for (int layer = 0; layer < numActivs.length - 1; layer++)
                        {
                           for (int left = 0; left < numActivs[layer]; left++)
                           {
                              for (int right = 0; right < numActivs[layer + 1]; right++)
                              {
                                 weights[layer][left][right] = (double)scan.nextInt();
                              }
                           }
                        }
                     }
                     else 
                     {
                        throw new RuntimeException("Invalid weight propagation style: \"" + weightPropStyle + "\"");
                     }
                  } // else clause where numActivs != null, inputs != null, and targets != null
               } //else if (label.equals("weights"))
               
               else if (label.equals("iterations"))
               {
                  maxIterations = Integer.parseInt(line.substring(colonIndex + 2));
                  printOutIteration = scan.nextInt();
               }
               else if (label.equals("errorThreshold"))
               {
                  errorThreshold = Double.parseDouble(line.substring(colonIndex + 2));
               }
               else if (label.equals("lambda"))
               {
                  lambda = Double.parseDouble(line.substring(colonIndex + 2));
               }
               else
               {
                  throw new RuntimeException("Unrecognized configuration: \"" + label + "\"");
               }
               
            } //if (colonIndex > -1) 
         } //if (!line.isEmpty())
      } //while (scan.hasNext())
      
      scan.close();
      
      //this.printNumPic(inputs[0]);

      System.out.println("DEBUG: end of config");
      
   } // public PerceptronConfigurer11(File file) throws RuntimeException, FileNotFoundException, IOException
   
   /**
    * Cleans up an image of RBG pels
    * First, this method converts the image to grayscale, then extracts the blue
    * pel value. Makes all pel values less than 100 to black (to decrease the effect of
    * the gray-ish whiteboard), then centers the image using center-of-mass calculations
    * (center of mass is abbreviated as CoM or com)
    * 
    * The magic number 10 was arbitrarily chosen since it appeared to deal with the whiteboard
    * relatively well (i.e. not masking too much hand but also masking most of the whiteboard).
    * 
    * @param pels the original set of image pels to convert 
    * @return the modified set of image pels 
    */
   private int[][] handProcess(int[][] pels)
   {
      double xcom = 0.0;
      double ycom = 0.0;
      double sumPels = 0.0; 
      
      DibDump1 dib = new DibDump1();
      
      // loop through all the given pels, convert to grayscale, extract blue, and find CoM
      for (int x = 0; x < pels.length; x++)
      {
         for (int y = 0; y < pels[0].length; y++)
         {
            int pel = pels[x][y];
            pel = ~(dib.colorToGrayscale(pel));
            
            pel = dib.pelToRGBQ(pel).red;
            
            if (pel < 100) {pel = 0;} // magic number 
            
            pels[x][y] = pel;
            
            //System.out.println("DEBUG pel hERE: " + pel);
            
            sumPels += pel;
            xcom += x * pel;
            ycom += y * pel;
            
         } // for (int y = 0; y < pels[0].length; y++)
      } // for (int x = 0; x < pels.length; x++)

      dib.writeOut(pels, "DEBUGgray.bmp");
      
      xcom /= sumPels;
      ycom /= sumPels;
      System.out.println("DEBUG\tsumpels: " + sumPels);
      
      int halfImage = 0;
      
      if (squareImage)
      {
         halfImage = pels.length / 2;
      }
      else
      {
         throw new RuntimeException("Cannot process non-square image.");
      }
      
      int xshift = halfImage - (int)xcom;
      int yshift = halfImage - (int)ycom;
      
      // shift all pels into a newpels array (to center )
      int[][] newpels = new int[pels.length][pels[0].length];
      
      for (int x = 0; x < newpels.length; x++)
      {
         for (int y = 0; y < newpels[0].length; y++)
         {
            int newx = x + xshift;
            int newy = y + yshift;
            boolean canShift = newx >= 0 && newx < newpels.length && newy >= 0 && newy < newpels[0].length;
            
            if(canShift)
            {
               newpels[newx][newy] = pels[x][y];
            }
         }
      }
      
      return newpels;
      
   } // private int[][] handProcess(int[][] pels)
   
   /**
    * Prints the values of a 50 by 50 array 
    * in a 50 by 50 square wall of numbers 
    * 
    * @param arr the array to be printed (MUST BE 50 BY 50)
    */
   public static void printNumPic(double[] arr)
   {
      int count = 0;
      for (int i = 0; i < 50; i++)
      {
         for (int j = 0; j < 50; j++)
         {
            System.out.printf("%.2f ", arr[count]);
            count++;
         }
         System.out.println();
      }
   }
   
   /**
    * Fills a given array of random weights generated from a given range
    * 
    * @param w8s the original array to fill with random weights
    * @param min the minimum value of random number generation
    * @param max the minimum value of random number generation
    * @return the updated weight array 
    */
   private double[][][] propWeightsRand(double[][][] w8s, double min, double max) 
   {
      for (int i = 0; i < w8s.length; i++) 
      {
         for (int j = 0; j < w8s[0].length; j++) 
         {
            for (int k = 0; k < w8s[0][0].length; k++) 
            {
               w8s[i][j][k] = random(min, max);
            }
         }
      }
      return w8s;
   }
   
   /**
    * Determines a random number within a given range
    * 
    * @param min minimum possible value
    * @param max maximum possible value 
    * @return a random number between the min and max values
    */
   private double random(double min, double max)
   {
      return Math.random() * (max - min) + min;
   }
   
   /**
    * Scales a true color pel value to the range [0,1]
    * Removes the stuffing of 1s into the empty bytes
    * 
    * @precondition the range of the pel values must match COLOR_BITS
    * 
    * @param n the true color pel value
    * @return a scaled number in the range [0,1]
    */
   private double scale(double n)
   {
      return ((double)((int)n & 0x00FFFFFF)) / numBits;
   }
   
   /**
    * Scales a number from [0,1] to a true color pel 
    * 
    * @param n a given number from [0,1]
    * @return a true color pel value in the range [0, COLOR_BITS]
    */
   public static double reverseScale(double n)
   {
      return n * numBits;
   }
   
} // public class PerceptronConfigurer11 