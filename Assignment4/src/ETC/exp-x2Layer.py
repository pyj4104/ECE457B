from pybrain.structure.modules.neuronlayer import NeuronLayer
from numpy import exp, dot, multiply, sqrt

class QuadraticPolynomialLayer(NeuronLayer):

	def _forwardImplementation(self, inbuf, outbuf):
		outbuf[:] = exp(multiply(-1, sqrt(inbuf)))

	def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
		inerr[:] = outerr * dot(multiply(x, -2), exp(multiply(-1, sqrt(x))))