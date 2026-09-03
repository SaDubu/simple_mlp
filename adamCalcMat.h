#pragma once

#include "cmatrix.h"

#include <vector>
#include <stddef.h>

//사용 안함.

//matrix를 사용하는 게산 등을 모두 여기서 담당
class CCalcc {
private:
	double alpha = 0;
	const double beta1 = 0.9;
	const double beta2 = 0.999;
	const double epslion = 1e-8;
	double loss = 0;

	size_t mSampleRowNum = 0;
public:
	CCalcc() {};

	CMat* leakyRelu(CMat* weight, CMat* mat);

	CMat* leakyReluDerivative(CMat* mat);

	CMat* softmax(CMat* mat);

	bool oneHotEncoding(CMat* mat);

	double categoricalCrossEntropy(CMat* target, CMat* predictedValue);

	CMat* calcGradient(CMat* weight, CMat* gradient = nullptr, CMat* hiddenLayer = nullptr);

	CMat* calcDerivativeFunction(CMat* target, CMat* preadictedMat);

	bool adam(std::vector<CMat*>* weight, std::vector<CMat*>* gradients);

	double calculateAccuracy(CMat* target, CMat* predictedValue);

	int predictedNumber(CMat* mat);

	CMat* forwardPropagation(CMat* inputLayer, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* weight);

	bool backwardPropagation(size_t repeatNum, CMat* input, CMat* output, CMat* target, std::vector<CMat*>* weight, std::vector<CMat*>* hiddenLayers);

	~CCalcc() {};
};