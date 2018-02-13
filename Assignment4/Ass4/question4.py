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

if __name__ == '__main__':
	main()
