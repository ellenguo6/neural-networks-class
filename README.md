# neural-networks-class

The fully connected feed-forward neural network I created in my Neural Networks class, built from scratch and implemented with arrays. 

Back propagation has been implemented for any number of hidden layers.

Sample configuration files are a.txt and d.txt. See READMEconfiguration for more details. 

a.txt: trains network for boolean functions. Two inputs and 3 outputs, 4 cases. Inputs are 0 or 1 (the four cases being each combination). The first output represents the result of the AND function, the second output the OR function, and the last output the XOR function. Two hidden layers, one with 4 nodes and the other with 5.  
b.txt: 
d.txt: trains network to recognize hand gestures. A bitmap image of a hand with one finger up will output 0, two fingers 0.2, three fingers 0.4, four fingers 0.6, and five fingers 0.8. 

Corresponding sample output files are loga.txt, logb.txt, and logd.txt. 
