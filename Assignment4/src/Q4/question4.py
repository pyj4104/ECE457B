from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from numpy import genfromtxt, zeros

def main():
	# Get Data
	dataSets = genfromtxt('normalizedData.csv', delimiter=',')
	alldata = ClassificationDataSet(13, 1, nb_classes=3)
	for dataSet in dataSets:
		alldata.addSample(dataSet[1:14], int(dataSet[0])-1)

	# Split the data
	tstdata_temp, trndata_temp = alldata.splitWithProportion(0.25)
	tstdata = ClassificationDataSet(13, 1, nb_classes=3)
	for n in range(0, tstdata_temp.getLength()):
		tstdata.addSample(tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1])
	trndata = ClassificationDataSet(13, 1, nb_classes=3)
	for n in range(0, trndata_temp.getLength()):
		trndata.addSample(trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1])
	trndata._convertToOneOfMany()
	tstdata._convertToOneOfMany()

	# Build Network
	fnn = buildNetwork(trndata.indim, 4, 4, 4, trndata.outdim)

	# Construct Trainer
	trainer = BackpropTrainer(fnn, trndata, learningrate=0.1)

	# Train
	while True:
		trainer.trainEpochs(1)
		trnresult = percentError(trainer.testOnClassData(), trndata['class'])
		print("Training Test Error: %5.2f%%" % trnresult)
		if trnresult < 1:
			break

	tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
	print("test error: %5.2f%%" % tstresult)
	out1 = fnn.activate([0.70789474,0.13636364,0.60962567,0.31443299,0.41304348,0.83448276,0.70253165,0.11320755,0.51419558,0.47098976,0.33333333,0.58608059,0.71825963])
	out2 = fnn.activate([0.26578947,0.70355731,0.54545455,0.58762887,0.10869565,0.3862069,0.29746835,0.54716981,0.29652997,0.11262799,0.25203252,0.47619048,0.21540656])
	out3 = fnn.activate([0.81578947,0.66403162,0.73796791,0.71649485,0.2826087,0.36896552,0.08860759,0.81132075,0.29652997,0.67576792,0.10569106,0.12087912,0.20114123])
	print(out1, out2, out3)

if __name__ == '__main__':
	main()
