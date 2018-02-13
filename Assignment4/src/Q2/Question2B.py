from pybrain.structure import LinearLayer, TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from numpy import exp, sqrt, multiply, arctan, linspace
from os.path import isfile

if __name__ == "__main__":
	net1Filename = "./Q2BA5.xml"
	net2Filename = "./Q2BB5.xml"
	trainingSetsA = linspace(0, 2, num=5)
	validationSetsA = linspace(0, 2, num=100)
	trainingSetsB = linspace(0, 1, num=5)
	validationSetsB = linspace(0, 1, num=100)
	a = lambda x: exp(multiply(-1, sqrt(x)))
	b = lambda x: arctan(x)
	network1 = buildNetwork(1, 5, 1) if not isfile(net1Filename) else NetworkReader.readFrom(net1Filename)
	network2 = buildNetwork(1, 5, 1) if not isfile(net2Filename) else NetworkReader.readFrom(net2Filename)
	if not isfile(net1Filename): NetworkWriter.writeToFile(network1, net1Filename) 
	if not isfile(net2Filename): NetworkWriter.writeToFile(network2, net2Filename)
	trainingSets1 = SupervisedDataSet(1, 1)
	trainingSets2 = SupervisedDataSet(1, 1)
	[trainingSets1.addSample(trainingSetsA[i], a(trainingSetsA[i])) for i in range(len(trainingSetsA))]
	[trainingSets2.addSample(trainingSetsB[i], b(trainingSetsB[i])) for i in range(len(trainingSetsB))]
	trainer1 = BackpropTrainer(network1, trainingSets1, learningrate = 0.1)
	trainer2 = BackpropTrainer(network2, trainingSets2, learningrate = 0.1)
	trainer1.trainUntilConvergence()
	trainer2.trainUntilConvergence()
	trainOutputA = []
	trainOutputB = []
	validateOutputA = []
	validateOutputB = []
	errA = []
	errB = []
	[trainOutputA.append(a(trainingSetsA[i])) for i in range(len(trainingSetsA))]
	[trainOutputB.append(b(trainingSetsB[i])) for i in range(len(trainingSetsB))]
	[validateOutputA.append(a(validationSetsA[i])) for i in range(len(validationSetsA))]
	[validateOutputB.append(b(validationSetsB[i])) for i in range(len(validationSetsB))]
	[print("Training A: (" + str(trainingSetsA[i]) + ", " + str(trainOutputA[i]) + ")") for i in range(len(trainingSetsA))]
	[print("Training B: (" + str(trainingSetsB[i]) + ", " + str(trainOutputB[i]) + ")") for i in range(len(trainingSetsB))]
	[print("Validation A: (" + str(validationSetsA[i]) + ", " + str(validateOutputA[i]) + ")") for i in range(len(validationSetsA))]
	[print("Validation B: (" + str(validationSetsB[i]) + ", " + str(validateOutputB[i]) + ")") for i in range(len(validationSetsB))]
	[errA.append(network1.activate([validationSetsA[i]])[0] - a(validationSetsA[i])) for i in range(len(validationSetsA))]
	[errB.append(network2.activate([validationSetsB[i]])[0] - b(validationSetsB[i])) for i in range(len(validationSetsB))]
	[print("The error of network a for " + str(i+1)+"th validation set is", errA[i]) for i in range(len(validationSetsA))]
	[print("The error of network b for " + str(i+1)+"th validation set is", errB[i]) for i in range(len(validationSetsB))]
	print("The total error for A is", sum(map(abs, errA)))
	print("The total error for B is", sum(map(abs, errB)))

