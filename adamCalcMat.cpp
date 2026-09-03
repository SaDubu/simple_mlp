#include "cmatrix.h"
#include "adamCalcMat.h"

#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <stddef.h>

//사용 안함.

CMat* CCalcc::leakyRelu(CMat* weight, CMat* mat) {
	//내적 결과
	CMat copyMat(mat);
	CMat productResult;

	productResult = copyMat * weight;

	// x > 0 ? x : 0.01 * x
	for (size_t r = 0; r < productResult.getRow(); r++) {
		for (size_t c = 0; c < productResult.getCol(); c++) {
			double x = productResult.getVal(r, c);
			productResult.changeVal(r, c,
				(x > 0 ? x : 0.01 * x));
		}
	}

	CMat* result = new CMat(&productResult);
	return result;
}

CMat* CCalcc::leakyReluDerivative(CMat* mat) {
	CMat* result = new CMat(mat);

	for (size_t r = 0; r < result->getRow(); r++) {
		for (size_t c = 0; c < result->getCol(); c++) {
			result->changeVal(r, c,
				(result->getVal(r, c) > 0 ? 1 : 0, 01));
		}
	}

	return result;
}

CMat* CCalcc::softmax(CMat* mat) {
	CMat* result = new CMat(mat);
	for (size_t r = 0; r < result->getRow(); r++) {
		double sum = 0;
		double maxVal = 0;
		//각 행의 최댓값 구하기
		for (size_t c = 0; c < result->getCol(); c++) {
			if (maxVal < result->getVal(r, c)) {
				maxVal = result->getVal(r, c);
			}
		}

		//지수화 e^n 꼴일 때 커짐을 방지하기 위해 최댓값을 빼줌
		for (size_t c = 0; c < result->getCol(); c++) {
			result->changeVal(r, c, std::exp(result->getVal(r, c) - maxVal));
			//분모를 구하는 과정
			sum += result->getVal(r, c);
		}

		//편차
		for (size_t c = 0; c < result->getCol(); c++) {
			result->changeVal(r, c,
				(result->getVal(r, c) / sum));
		}
	}

	return result;
}
//항목 별 오차 사용
double CCalcc::categoricalCrossEntropy(CMat* target, CMat* predictedValue) {

	loss = 0; //초기화

	for (size_t r = 0; r < target->getRow(); r++) {

		for (size_t c = 0; c < target->getCol(); c++) {
			loss += target->getVal(r, c) * std::log(predictedValue->getVal(r, c));
		}

	}

	loss /= target->getRow();

	if (loss < 0) return -loss;

	return loss;
}
bool CCalcc::adam(std::vector<CMat*>* weight, std::vector<CMat*>* gradients) {
	

	return true;
}
//기울기 계산
CMat* CCalcc::calcGradient(CMat* weight, CMat* gradient, CMat* hiddenLayer) {
	if (gradient == nullptr) {
		return weight;
	}
	std::vector<double> insertMat;

	size_t row = weight->getRow();
	size_t col = weight->getCol();
	CMat hiddenRowTarget;

	CMat copyMat(1, 1);

	hiddenRowTarget = hiddenLayer->getRowVal(mSampleRowNum);

	CMat copyHiddenLayerRowVal;

	copyHiddenLayerRowVal = hiddenLayer->getRowVal(mSampleRowNum);

	copyMat = leakyReluDerivative(&copyHiddenLayerRowVal);
	for (size_t r = 0; r < row; r++) {
		double sum = 0;
		for (size_t c = 0; c < col; c++) {
			sum += gradient->getVal(0, c) * weight->getVal(r, c);
		}
		sum = sum * copyMat.getVal(0, r);
		insertMat.emplace_back(sum);
	}

	CMat* result = new CMat(1, insertMat.size());
	for (size_t c = 0; c < result->getCol(); c++) {
		result->changeVal(0, c, insertMat[c]);
	}
	insertMat.clear();
	std::vector<double>().swap(insertMat);
	return result;
}
//도함수 계산
CMat* CCalcc::calcDerivativeFunction(CMat* target, CMat* predictedMat) {
	CMat error;
	CMat copyPredictedMat(predictedMat);
	CMat copyErrorRow;
	CMat derivativedMat;
	CMat copyPredictedMatRowVal;

	copyPredictedMatRowVal = copyPredictedMat.getRowVal(mSampleRowNum);

	error = copyPredictedMat - target;
	derivativedMat = leakyReluDerivative(&copyPredictedMatRowVal);

	copyErrorRow = error.getRowVal(mSampleRowNum);
	copyErrorRow.convertUnitMat();
	copyErrorRow = copyErrorRow * &derivativedMat;
	copyErrorRow = copyErrorRow.transpose();

	CMat* result = new CMat(&copyErrorRow);
	return result;
}
//원핫인코딩
bool CCalcc::oneHotEncoding(CMat* mat) {
	const size_t oneHotMatCol = 10;
	CMat copyMat(mat);
	if (copyMat.getCol() != 1) {
		return false;
	}
	mat->resize(copyMat.getRow(), oneHotMatCol);

	for (size_t r = 0; r < mat->getRow(); r++) {
		mat->changeVal(r, (size_t)copyMat.getVal(r, 0), 1);
	}

	return true;
}
//epoch당 평균 정확도
double CCalcc::calculateAccuracy(CMat* target, CMat* predictedValue) {
	double result = 0;
	double count = 0;
	CMat targetMat;
	CMat predicteMat;
	if (target->getRow() != predictedValue->getRow() && target->getCol() != predictedValue->getCol()) {
		return result;
	}
	for (size_t r = 0; r < target->getRow(); r++) {
		targetMat = target->getRowVal(r);
		predicteMat = predictedValue->getRowVal(r);
		if (predictedNumber(&targetMat) == predictedNumber(&predicteMat)) {
			count++;
		}
	}

	result = count / target->getRow();
	return result;
}
//출력 matrix이랑 label matrix의 최대값(정답값 or 예측값) 찾아 int형태로 반환함
int CCalcc::predictedNumber(CMat* mat) {
	int result = -1;
	double maxVal = 0;

	for (size_t r = 0; r < mat->getRow(); r++) {
		for (size_t c = 0; c < mat->getCol(); c++) {
			if (maxVal < mat->getVal(r, c)) {
				maxVal = mat->getVal(r, c);
				result = (int)c;
			}
		}
	}

	return result;
}
//forward Propagation
CMat* CCalcc::forwardPropagation(CMat* inputLayer, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* weight) {
	//vector들의 size를 검사하여서 1번째 가중치 벡터와 입력을 relu하고 hidden 첫번째에 저장 그 다음엔 hidden 1과 weight2를 가지고 그 다음 hidden2나 return하기
	if (weight->size() <= 0) {
		return inputLayer;
	}
	hiddenLayers->emplace_back(leakyRelu(weight->at(0), inputLayer));
	delete hiddenLayers->at(0);
	hiddenLayers->erase(hiddenLayers->begin());

	size_t nonChageValue = hiddenLayers->size() - 1;
	for (size_t index = 1; index < weight->size() - 1; index++) {
		hiddenLayers->emplace_back(leakyRelu(weight->at(index), hiddenLayers->at(nonChageValue)));
		//제일 앞을 지워야함.
		delete hiddenLayers->at(0);
		hiddenLayers->erase(hiddenLayers->begin());
	}

	if (weight->size() == 1) {
		return softmax(hiddenLayers->at(0));
	}

	return softmax(leakyRelu(weight->at(weight->size() - 1), hiddenLayers->at(hiddenLayers->size() - 1)));
}
//Back Propagation
bool CCalcc::backwardPropagation(size_t repeatNum, CMat* input, CMat* output, CMat* target, std::vector<CMat*>* weight, std::vector<CMat*>* hiddenLayers) {
	std::random_device random;

	std::mt19937 gen(random());

	std::uniform_int_distribution<size_t> randomNum(0, (input->getRow() - 1));

	bool isDone = true;

	mSampleRowNum = 0;

	if (input->getRow() > 1) {
		mSampleRowNum = randomNum(gen);
	}

	CMat derivativeFunction;

	derivativeFunction = calcDerivativeFunction(target, output);

	std::vector<CMat*> gradients;
	CMat gradient;

	gradients.emplace_back(new CMat(derivativeFunction));

	size_t hiddenLayersSize = hiddenLayers->size();
	for (size_t index = weight->size() - 1; index > 0; index--) {
		if (hiddenLayersSize != 0) {
			hiddenLayersSize--;
		}
		gradient = calcGradient(weight->at(index), gradients.at(gradients.size() - 1), hiddenLayers->at(hiddenLayersSize));
		gradients.emplace_back(new CMat(&gradient));
	}

	for (size_t index = 0; index < gradients.size(); index++) {
		delete gradients.at(index);
	}

	std::vector<CMat*>().swap(gradients);
	return isDone;
}