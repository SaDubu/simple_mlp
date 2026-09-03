#pragma once

#include "cmatrix.h"
#include <vector>
#include <stddef.h>
#include <string>

//matrix를 사용하는 게산 등을 모두 여기서 담당
class CCalc {
private:
	double learningRate = 0.01;
	double loss = 0;
	double dropRate = 0.2;

	std::string activeName;
	std::string optimizerName;

	size_t mSampleRowNum = 0;
public:
	CCalc() {};

	CMat* relu(CMat* weight, CMat* mat, CMat* bias = nullptr);

	CMat* leakyRelu(CMat* weight, CMat* mat, CMat* bias = nullptr);

	CMat* reluDerivative(CMat* mat);

	CMat* leakyReluDerivative(CMat* mat);

	CMat* softmax(CMat* mat);

	bool oneHotEncoding(CMat* mat);

	double categoricalCrossEntropy(CMat* target, CMat* predictedValue);

	double meanSquareError(CMat* target, CMat* predictedValue);

	CMat* calcGradient(CMat* weight, CMat* gradient = nullptr, CMat* hiddenLayer = nullptr);

	CMat* calcDerivativeFunction(CMat* target, CMat* preadictedMat);

	bool gradinetDescent(CMat* input, CMat* derivativeFuction, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers, int repeatNum = 0);

	bool updateWeight(CMat* input, std::vector<CMat*>* gradients, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers);

	CMat* generalizeMatrix(CMat* mat);

	double calculateAccuracy(CMat* target, CMat* predictedValue);

	int predictedNumber(CMat* mat);

	//test
	void fastMul(CMat* input, std::vector<CMat*>* gradients, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers);

	void upadteBias(std::vector<CMat*>* biases, std::vector<CMat*>* gradients, std::vector<CMat*>* weight);

	void setLearningRate(double val) {
		learningRate = val;
	}

	double getLearningRate() {
		return learningRate;
	}

	void setDropRate(int val) {
		dropRate = (double)val * 0.01;
	}

	double getDropRate() {
		return dropRate;
	}

	void dropOut(CMat* hiddenLayer);

	CMat* elu(CMat* weight, CMat* mat, CMat* bias = nullptr);

	CMat* eluDerivative(CMat* mat);

	double sigmoid(double x);

	double sigmoidDerivative(double x);

	CMat* silu(CMat* weight, CMat* mat, CMat* bias = nullptr);

	CMat* siluDerivative(CMat* mat);

	void adam(std::vector<CMat*>* gradient, std::vector<CMat*>* weights, int repeatNum);

	void setAdamP(std::vector<CMat*>* weights );

	void initP(std::vector<CMat*>* vec);

	CMat* selectActive(CMat* weight, CMat* mat, CMat* bias = nullptr);

	CMat* setD(CMat* mat);

	void setActName(std::string activeName);

	void setOptimizer(std::string opttimizerName);
	//

	CMat* testForward(CMat* inputLayer, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* weight, std::vector<CMat*>* biases);

	CMat* forwardPropagation(CMat* inputLayer, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* weight, std::vector<CMat*>* biases, bool onOff = false);

	bool backwardPropagation(size_t repeatNum, CMat* input, CMat* output, CMat* target, std::vector<CMat*>* weight, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* biases);

	~CCalc() {};
};