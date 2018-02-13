from pybrain.structure import LinearLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from numpy import exp, multiply, arctan, linspace, mgrid, square
from math import pi, sin, cos
from os.path import isfile

def withInLimits(dataSets, function, network, goalAvgErr):
	err = meanSquareError(dataSets, function, network)
	print("The average error was", err)
	return True if err < goalAvgErr else False

def meanSquareError(dataSets, function, network):
	avgErr = 0
	for dataSet in dataSets:
		avgErr += square(network.activate(dataSet) - function(dataSet))
	return avgErr/len(dataSets)

if __name__ == "__main__":
	# Initialize the network
	#networkName = "./q35.xml"
	network = buildNetwork(2, 50, 1, outputbias=False, bias=False) #if not isfile(networkName) else NetworkReader.readFrom(networkName)
	#if not isfile(networkName): NetworkWriter.writeToFile(network, networkName)

	# Make data sets
	training = mgrid[-1:1.1:2/5, -1:1.1:2/10].reshape(2,-1).T
	validation = mgrid[-1:1.1:2/10, -1:1.1:2/5].reshape(2,-1).T

	# Declare mapping function
	a = lambda x: sin(2*pi*x[0]) * cos(0.5*pi*x[1])
	
	# Train
	trainingSets = SupervisedDataSet(2, 1)
	[trainingSets.addSample(training[i], a(training[i])) for i in range(len(training))]
	trainer = BackpropTrainer(network, trainingSets, learningrate = 0.01)
	i = 0
	while not(withInLimits(training, a, network, 0.2)):
		print("Training the network for "+str(i+1)+"th time. ", end="")
		i += 1
		trainer.trainEpochs()

	#trainOut = []
	#validateOut = []
	#err = []
	#[trainOut.append(a(training[i])) for i in range(len(training))]
	#[validateOut.append(a(validation[i])) for i in range(len(validation))]
	#[print("Training A: (" + str(trainingSetsA[i]) + ", " + str(trainOutputA[i]) + ")") for i in range(len(trainingSetsA))]
	#[print("Training B: (" + str(trainingSetsB[i]) + ", " + str(trainOutputB[i]) + ")") for i in range(len(trainingSetsB))]
	#[print("Validation A: (" + str(validationSetsA[i]) + ", " + str(validateOutputA[i]) + ")") for i in range(len(validationSetsA))]
	#[print("Validation B: (" + str(validationSetsB[i]) + ", " + str(validateOutputB[i]) + ")") for i in range(len(validationSetsB))]
	#[err.append(network.activate(validation[i])[0] - a(validation[i])) for i in range(len(validation))]
	#[print("The error of network a for " + str(i+1)+"th validation set is", errA[i]) for i in range(len(validationSetsA))]
	#[print("The error of network b for " + str(i+1)+"th validation set is", errB[i]) for i in range(len(validationSetsB))]
	print("The total error for A is", meanSquareError(validation, a, network))

