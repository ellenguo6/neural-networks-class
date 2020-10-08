package neuralnets;
import java.util.Scanner;
import java.io.BufferedWriter;
import java.io.File; 
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException; 

/**
 * A tester class that creates the Perceptron by getting a user-defined 
 * configuration file and training the network. 
 * 
 * @author Ellen Guo
 * @version 10 January 2020
 *
 */
public class PerceptronTester11 
{  
   /**
    * Main method that creates a neural net and trains it. 
    * 
    * @param args from the command line
    * @throws FileNotFoundException
    * @throws IOException BufferedWriter (output log file writing) failure
    */
   public static void main(String[] args) throws FileNotFoundException, IOException
   {
      Scanner fileNameGetter = new Scanner(System.in);
      System.out.println("Enter the name of the configuration file: ");
      String fileName = fileNameGetter.nextLine();
      fileNameGetter.close();
      
      File file = new File(fileName);

      PerceptronConfigurer11 config = new PerceptronConfigurer11(file);
      Perceptron11 network = new Perceptron11(config, "log" + fileName);
      network.run();
      
      System.out.println("Done.");
   } // public static void main(String[] args) throws FileNotFoundException, IOException
   
} // public class PerceptronTester11 