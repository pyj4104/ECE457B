import os

fileName = "./dataFile.txt"
outputFileName = "./normalizedData.csv"

def writeToNewFile(newLines):
	with open(outputFileName, "w") as f:
		for newLine in newLines:
			for i in range(len(newLine)):
				f.write(str(newLine[i]))
				if not i == (len(newLine) - 1):
					f.write(",")
			f.write("\n")

def normalize(newLines):
	for i in range(1,14):
		minimum = 10000000;
		maximum = 0;
		for j in range(len(newLines)):
			minimum = min(newLines[j][i], minimum)
			maximum = max(newLines[j][i], maximum)
		print(minimum, maximum)
		for j in range(len(newLines)):
			newLines[j][i] = round((newLines[j][i] - minimum)/(maximum - minimum), 8)
	return newLines


def readAndFormat():
	newLines = []
	with open(fileName) as f:
		content = f.read().split('\n')
		for line in content:
			formatedLine = []
			[formatedLine.append(float(intVal)) for intVal in line.split(",")]
			newLines.append(formatedLine)
			print(formatedLine)
	return newLines

def main():
	newLines = readAndFormat()
	newLines = normalize(newLines)
	[print(newLine) for newLine in newLines]
	writeToNewFile(newLines)

if __name__ == '__main__':
	main()