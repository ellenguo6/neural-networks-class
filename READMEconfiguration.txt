READ ME FOR PERCEPTRON CONFIGURATION FILE

numTestCases: the number of test cases (let's call this n)

numHiddenLayers: the number of hidden layers (let's call this h)
the next h integers represent the number of activations in the hth hidden layer

bitmap: yes or no answer

yes: 
yes indicates that the inputs will be the pel values extracted from bitmap images
The next line must contain either "grayscale" or "RGB"; this indicates the range of values per pel, which is used in scaling calculations.
The next line must contain either "square" or "not square"; this indicates if the bitmap images are square or not. If there are multiple bitmaps, they all must be square for this value to be "square"
The next line must contain either "image" or "number"; this indicates the type of output we want. "image" means that we would like the output to be another bitmap of the same size as the input one. This setting is used for image compression. "number" means that we would like to have just 1 numerical output. This setting is used for image classification (for the hand thing). 
The next line must contain either "hand" or "letter"; if "hand" is chosen, the image will be centered, and the whiteboard will be turned black.
The next n lines contain the filenames for the images to be used as inputs. Each one functions as its own test case. 
If the setting was "number", the next n lines must contain the target outputs for each test case.

no (to the bitmap response): 
no indicates that the inputs and outputs will be listed in subsequent lines as numerical values. 
The next value represents the number of inputs per test case (must be the same across all test cases) (let's call this i). 
The next n lines (each with i numbers) represent the i inputs of each nth test case
(skip a line)
The next value represents the number of outputs per test case (must be the same across all test cases) (let's call this p).
The next n lines (each with p numbers) represent the p inputs of each nth test case

weights: must be followed by the string "Random" or "Manual" (other formats are not accepted) to decide how weights are initially generated
If "Random," the next two numbers represent the min and max that bound the range for random weight generation, respectively.
If "Manual," the following list of weights represents the initial weight values (first one is w[0][0][0], second one is w[0][0][1], third is w[0][1][0], and so on and so forth, where w[layer][left][right])

iterations: the maximum number of iterations allowed before timeout 
The value on the next line (let's call this x) represents the print out iteration. That is, every x iterations, diagnostic information will be printed to the console and/or a bitmap will be generated. 

errorThreshold: threshold that the error of all test cases must be below before successful termination

lambda: multiplicative factor in gradient descent
