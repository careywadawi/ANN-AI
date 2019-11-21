from numpy import exp, array, dot

from read import normalized

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
          
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            layer1_error = layer2_delta.dot(self.layer2.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            self.layer1 += layer1_adjustment
            self.layer2 += layer2_adjustment


    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2))
        return output_from_layer1, output_from_layer2


    def print_weights(self):
        print(self.layer1)
        print(self.layer2)


if __name__ == "__main__":
  
    layer1 = array([[0.2, 0.1], [0.3, 0.1], [0.2, 0.1]])

    layer2 = array([[0.5, 0.1]]).T

    neural_network = NeuralNetwork(layer1, layer2)

    neural_network.print_weights()

    training_set_inputs = array(
        [
            [normalized_set['input1'][0], normalized_set['input2'][0], normalized_set['input3'][0]],
            [normalized_set['input1'][1], normalized_set['input2'][1], normalized_set['input3'][1]],
            [normalized_set['input1'][2], normalized_set['input2'][2], normalized_set['input3'][2]],
            [normalized_set['input1'][3], normalized_set['input2'][3], normalized_set['input3'][3]],
            [normalized_set['input1'][4], normalized_set['input2'][4], normalized_set['input3'][4]],
            [normalized_set['input1'][5], normalized_set['input2'][5], normalized_set['input3'][5]]
        ])

    training_set_outputs = array(
        [[
            normalized_set['output'][0],
            normalized_set['output'][1],
            normalized_set['output'][2],
            normalized_set['output'][3],
            normalized_set['output'][4],
            normalized_set['output'][5]
        ]]).T

    print("Inputs", training_set_inputs)
    print("Output", training_set_outputs)

    neural_network.train(training_set_inputs, training_set_outputs, 60000)

 
    print("Weights ")
    neural_network.print_weights()

  
    output = neural_network.think(array([0.5, 0.6, 0.1]))
    print("Weights", output[0])
    print("Out ", output[1])

   