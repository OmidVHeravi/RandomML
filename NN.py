
# import neccessary Dependancies/libs
import numpy as np
import pandas as pd
from dataPrep import features, targets, features_test, targets_test
import math

#Init Perceptron class
class Perceptron():

    """
    One layer Perceptron example
    """

    #initialize with  weights, threshold, and bias
    def __init__(self, threshold=0, bias=1, learnrate=0.1, target=[1,1,1], Inputs=[1,1,1], epochs=10000, weights=[1,1,1]):
        """
        Weights matrix/values, must equal the first neuron layer in length, can be n-dimensional in length, default value of [1,1,1] with shape of (3,1)
        """
        self.weights = weights
        self.threshold = threshold
        self.bias = bias
        self.learnrate = learnrate
        self.target = target
        self.Inputs = Inputs
        self.epochs = epochs


    def sigmoid(self,x):
        """
        Sigmoid activation/step function. It takes in @param Inputs, as the value of n-dimensional first layer of the neural network. For its input, it computes the dot product of the @param valueOfFirstLayer and the weights plus the bias. The output is a ratio in between 0 and 1 (float value).
        """
        #output = np.dot(self.weights,Inputs) + self.bias
        activatedOutput = (1 / (1 + np.exp(-x)))
        return activatedOutput

    def prime_sigmoid(self, Inputs):
        """
        fucntion to calculate the derivative of our sigmoid of the inputs.
        """
        return self.sigmoid(Inputs) * ( 1 - self.sigmoid(Inputs))

    def tanh(self,x):
        tanh_output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return tanh_output

    def tanh_prime(self, x):
        prime_tanh_output = (1 - math.tanh(x)**2)
        return prime_tanh_output


    def gradient_Step(self):

        # Use to same seed to make debugging easier
        #np.random.seed(42)

        n_records, n_features = features.shape
        last_loss = None

        # Initialize weights
        weights = np.random.normal(scale= 1 / n_features**.5, size=n_features)

        #repeat the processs of tunning weights as many as e-times
        for e in range(self.epochs):

            #change in weights, which will be updated after each calculation below
            del_w = np.zeros(weights.shape)

            for x,y in zip(features.values,targets): #iterate thorugh each records, x is input, y is target

                h = np.dot(x, weights) # h is the calculated dot product of one row from features, x with weights
                predicted_output = self.sigmoid(h) #predict the y for the input x, with the current weights

                error = y - predicted_output #Calculate the error, real value y - predicted_output
                derivative_of_h = self.prime_sigmoid(h) #Calculate the rate of change of the h using sigmoid_prime function

                real_error_term = error * derivative_of_h #Calculate the real error term using multiplicaiton of error and derivative_of_h
                del_w += real_error_term * x #calculate the change for the delta weight by multiplying the x row by the error term

            #print("New delta weights is {}".format(del_w))
            weights += self.learnrate * del_w / n_records # update the global weights by multiplying learnrate times the delta_Weight divided by the amount of terms in the training set
            print(weights)

            if e % (self.epochs / 10) == 0: #printing out the mean square error on the training set
                out = self.sigmoid(np.dot(features,weights))
                loss = np.mean((out - targets) ** 2)
                if last_loss and last_loss < loss:
                    print("Train loss: ", loss, "  WARNING - Loss Increasing")
                else:
                    print("Train loss: ", loss)
                last_loss = loss

        # Calculate accuracy on test data
        tes_out = self.sigmoid(np.dot(features_test, weights))
        predictions = tes_out > 0.5
        accuracy = np.mean(predictions == targets_test)
        print("Prediction accuracy: {:.3f}".format(accuracy))


if __name__ == "__main__":

    p1 = Perceptron()
    print(p1.gradient_Step())
