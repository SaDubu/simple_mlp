#include "cmatrix.h"
#include "calcmat.h"

#include <cmath>
#include <random>
#include <limits>
#include <algorithm>
#include <stddef.h>
#include <iostream>

//matrix로 진행되는 모든 연산을 여기에 정의함(relu, softmax, loss, forward, backward 등)

//test

//drop out 구현해보기

void CCalc::dropOut(CMat* hiddenLayer) {
	int size = hiddenLayer->getCol();

	std::vector<int> dropIndex;

	for (int index = 0; index < size; index++) {
		dropIndex.emplace_back(index);
	}
	
	for (size_t row = 0; row < hiddenLayer->getRow(); row++) {	
		std::random_shuffle(dropIndex.begin(), dropIndex.end());

		int dropSize = size * dropRate;

		for (int index = 0; index < dropSize; index++) {
			hiddenLayer->changeVal(row, (size_t)dropIndex[index], 0.0);
		}
	}
}

//silu 함수 구현해보기

double CCalc::sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

double CCalc::sigmoidDerivative(double x) {
	return sigmoid(x) * (1.0 - sigmoid(x));
}

CMat* CCalc::silu(CMat* weight, CMat* mat, CMat* bias) {
	CMat copyMat(mat);
	CMat productResult;

	productResult = copyMat * weight;
	if (bias != nullptr) {
		productResult = productResult + bias;
	}

	for (size_t r = 0; r < productResult.getRow(); r++) {
		for (size_t c = 0; c < productResult.getCol(); c++) {
			double x = productResult.getVal(r, c);
			productResult.changeVal(r, c,
				(x * sigmoid(x)));
		}
	}

	CMat* result = new CMat(&productResult);
	return result;
}

CMat* CCalc::siluDerivative(CMat* mat) {
	CMat* result = new CMat(mat);

	for (size_t r = 0; r < result->getRow(); r++) {
		for (size_t c = 0; c < result->getCol(); c++) {
			double x = result->getVal(r, c);
			double sigmoidX = sigmoid(x);
			result->changeVal(r, c,
				(sigmoidX + x * sigmoidDerivative(x)));
		}
	}

	return result;
}
//elu 함수
CMat* CCalc::elu(CMat* weight, CMat* mat, CMat* bias) {
	//내적 결과
	CMat copyMat(mat);
	CMat productResult;

	productResult = copyMat * weight;
	if (bias != nullptr) {
		productResult = productResult + bias;
	}

	// x > 0 ? x : 0
	for (size_t r = 0; r < productResult.getRow(); r++) {
		for (size_t c = 0; c < productResult.getCol(); c++) {
			double x = productResult.getVal(r, c);
			productResult.changeVal(r, c,
				(x > 0 ? x : std::exp(x) - 1));//(x > 0 ? x : 0));
		}
	}

	CMat* result = new CMat(&productResult);
	return result;
}

CMat* CCalc::eluDerivative(CMat* mat) {
	CMat* result = new CMat(mat);

	for (size_t r = 0; r < result->getRow(); r++) {
		for (size_t c = 0; c < result->getCol(); c++) {
			result->changeVal(r, c,
				(result->getVal(r, c) > 0 ? 1 : result->getVal(r, c) + 1));//(result->getVal(r, c) > 0 ? 1 : 0));
		}
	}

	return result;
}

//adam 구현

std::vector<CMat*> m;
std::vector<CMat*> v;

const double beta = 0.9;
const double gama = 0.999;
const double epslion = 1e-8;

//솔직히 이거 뭐가 문제인지 생각이 전혀 나지 않음ㅋ
void CCalc::adam(std::vector<CMat*>* gradients, std::vector<CMat*>* weights, int repeatNum) {
	std::vector<CMat*> vHat, mHat;
	CMat gradient(1, 1);

	int gradientNum = gradients->size() - 1;

	for (int gIndex = gradientNum; gIndex >= 0; gIndex--) {
		gradient = (gradients->at(gIndex));
		gradient = gradient.transpose();
		{
			int index = 0;
			CMat mat(m.at(index));
			CMat g(&gradient);
			for (size_t row = 0; row < mat.getRow(); row++) {
				for (size_t col = 0; col < mat.getCol(); col++) {
					mat.getModifiableVal()[row][col] = beta * mat.getVal(row, col) +
						(1 - beta) * g.getVal(row, col);
				}
			}
			delete m.at(0);
			m.erase(m.begin());
			m.emplace_back(new CMat(&mat));
			mat.multiplyByScalar(1.0 / (1.0 - std::pow(beta, repeatNum)));
			mHat.emplace_back(new CMat(&mat));
		}

		{
			int index = 0;;
			CMat mat(v.at(index));
			CMat g(&gradient);
			for (size_t row = 0; row < mat.getRow(); row++) {
				for (size_t col = 0; col < mat.getCol(); col++) {
					mat.getModifiableVal()[row][col] = gama * mat.getVal(row, col) +
						(1 - gama) * std::pow(g.getVal(row, col), 2);
				}
			}
			delete v.at(0);
			v.erase(v.begin());
			v.emplace_back(new CMat(&mat));
			mat.multiplyByScalar(1.0 / (1.0 - std::pow(gama, repeatNum)));
			vHat.emplace_back(new CMat(&mat));
		}
	}

	for (int index = 0; index < weights->size(); index++) {
		CMat mat(weights->at(0));
		for (size_t row = 0; row < mat.getRow(); row++) {
			for (size_t col = 0; col < mat.getCol(); col++) {
				double x = learningRate * (mHat.at(index)->getVal(row, col) /
					std::sqrt(vHat.at(index)->getVal(row, col) + epslion));
				mat.getModifiableVal()[row][col] -= x;
			}
		}
		weights->emplace_back(new CMat(&mat));
		delete weights->at(0);
		weights->erase(weights->begin());
	}

	initP(&mHat);
	initP(&vHat);
}

void CCalc::setAdamP(std::vector<CMat*>* weights) {
	size_t weightNum = weights->size();

	if (m.size() > weightNum) {
		initP(&m);
	}
	
	if (v.size() > weightNum) {
		initP(&v);
	}


	if (m.size() == v.size() && m.size() != weightNum) {
		for (size_t index = 0; index < weightNum; index++) {
			CMat mat(weights->at(index)->getRow(), weights->at(index)->getCol());
			m.emplace_back(new CMat(&mat));
			v.emplace_back(new CMat(&mat));
		}
	}
}

void CCalc::initP(std::vector<CMat*>* vec) {
	for (CMat* mat : *vec) {
		delete mat;
	}

	std::vector<CMat*>().swap(*vec);
}

//활성화 함수 선택
std::string activeArray[4] = {
	"relu",
	"leakyRelu",
	"silu",
	"elu"
};
CMat* CCalc::selectActive(CMat* weight, CMat* mat, CMat* bias) {
	int choiced = -1;

	for (int index = 0; index < activeArray->size(); index++) {
		if (activeName.compare(activeArray[index])) {
			choiced = index;
			break;
		}
	}


	switch (choiced) {
	case 0:
		return relu(weight, mat, bias);
	case 1:
		return leakyRelu(weight, mat, bias);
	case 2:
		return silu(weight, mat, bias);
	case 3:
		return elu(weight, mat, bias);
	default:
		exit(1);
		return nullptr;
	}

	return nullptr;
}

CMat* CCalc::setD(CMat* mat) {
	int choiced = -1;

	for (int index = 0; index < activeArray->size(); index++) {
		if (activeName.compare(activeArray[index])) {
			choiced = index;
			break;
		}
	}

	switch (choiced) {
	case 0:
		return reluDerivative(mat);
	case 1:
		return leakyReluDerivative(mat);
	case 2:
		return siluDerivative(mat);
	case 3:
		return eluDerivative(mat);
	default:
		exit(1);
		return nullptr;
	}

	return nullptr;
}

void CCalc::setActName(std::string text) {
	activeName = text;
}

void CCalc::setOptimizer(std::string text) {
	optimizerName = text;
}

//test

CMat* CCalc::relu(CMat* weight, CMat* mat, CMat* bias) {
	//내적 결과
	CMat copyMat(mat);
	CMat productResult;

	productResult = copyMat * weight;
	if (bias != nullptr) {
		productResult = productResult + bias;
	}

	// x > 0 ? x : 0
	for (size_t r = 0; r < productResult.getRow(); r++) {
		for (size_t c = 0; c < productResult.getCol(); c++) {
			double x = productResult.getVal(r, c);
			productResult.changeVal(r, c,
				(x > 0 ? x : 0));
		}
	}

	CMat* result = new CMat(&productResult);
	return result;
}

CMat* CCalc::leakyRelu(CMat* weight, CMat* mat, CMat* bias) {
	//내적 결과
	CMat copyMat(mat);
	CMat productResult;

	productResult = copyMat * weight;
	if (bias != nullptr) {
		productResult = productResult + bias;
	}

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

CMat* CCalc::reluDerivative(CMat* mat) {
	CMat* result = new CMat(mat);

	for (size_t r = 0; r < result->getRow(); r++) {
		for (size_t c = 0; c < result->getCol(); c++) {
			result->changeVal(r, c,
				(result->getVal(r, c) > 0 ? 1 : 0));
		}
	}

	return result;
}

CMat* CCalc::leakyReluDerivative(CMat* mat) {
	CMat* result = new CMat(mat);

	for (size_t r = 0; r < result->getRow(); r++) {
		for (size_t c = 0; c < result->getCol(); c++) {
			result->changeVal(r, c,
				(result->getVal(r, c) > 0 ? 1 : 0.01));
		}
	}

	return result;
}

CMat* CCalc::softmax(CMat* mat) {
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
double CCalc::categoricalCrossEntropy(CMat* target, CMat* predictedValue) {

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
//평균 제곱 오차
double CCalc::meanSquareError(CMat* target, CMat* predictedValue) {

	loss = 0;

	for (size_t r = 0; r < target->getRow(); r++) {

		for (size_t c = 0; c < target->getCol(); c++) {
			loss += std::pow((target->getVal(r, c) - predictedValue->getVal(r, c)), 2);
		}

	}

	loss /= target->getCol();

	loss /= target->getRow();

	if (loss < 0) return -loss;

	return loss;
}
//기울기 계산
CMat* CCalc::calcGradient(CMat* weight, CMat* gradient, CMat* hiddenLayer) {
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

	copyMat = setD(&copyHiddenLayerRowVal);
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
CMat* CCalc::calcDerivativeFunction(CMat* target, CMat* predictedMat) {
	CMat error;
	CMat copyPredictedMat(predictedMat);
	CMat copyErrorRow;
	CMat derivativedMat;
	CMat copyPredictedMatRowVal;

	copyPredictedMatRowVal = copyPredictedMat.getRowVal(mSampleRowNum);

	error = copyPredictedMat - target;
	derivativedMat = setD(&copyPredictedMatRowVal);

	copyErrorRow = error.getRowVal(mSampleRowNum);
	copyErrorRow.convertUnitMat();
	copyErrorRow = copyErrorRow * &derivativedMat;
	copyErrorRow = copyErrorRow.transpose();

	CMat* result = new CMat(&copyErrorRow);
	return result;
}
//경사하강법
bool CCalc::gradinetDescent(CMat* input, CMat* derivativeFuction, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers, int repeatNum) {
	std::vector<CMat*> gradients;
	CMat gradient;

	gradients.emplace_back(new CMat(derivativeFuction));

	size_t hiddenLayersSize = hiddenLayers->size();
	for (size_t index = weight->size() - 1; index > 0; index--) {
		if (hiddenLayersSize != 0) {
			hiddenLayersSize--;
		}
		gradient = calcGradient(weight->at(index), gradients.at(gradients.size() - 1), hiddenLayers->at(hiddenLayersSize));
		gradients.emplace_back(new CMat(&gradient));
	}

	if (optimizerName.compare("adam")) {
		fastMul(input, &gradients, weight, biases, hiddenLayers);
		setAdamP(weight);
		adam(&gradients, weight, repeatNum);

		std::vector<CMat*>().swap(gradients);
		return true;
	}

	bool isDone = updateWeight(input, &gradients, weight, biases, hiddenLayers);

	if (!isDone) {
		return false;
	}

	std::vector<CMat*>().swap(gradients);

	return true;
}
//matrix의 row 값 간의 덧셈
CMat* CCalc::generalizeMatrix(CMat* mat) {
	CMat copyResult;
	CMat matRowVal;

	double mulVal = 1.0 / mat->getRow();
	copyResult = mat->getRowVal(0);
	for (size_t row = 1; row < mat->getRow(); row++) {
		matRowVal = mat->getRowVal(row);
		copyResult = copyResult + &matRowVal;
	}
	copyResult.multiplyByScalar(mulVal);

	CMat* result = new CMat(&copyResult);
	return result;
}
//test fast
void CCalc::fastMul(CMat* input, std::vector<CMat*>* gradients, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers) {
	CMat gradient;
	size_t hiddenLayersSize = hiddenLayers->size();
	CMat copyMat;
	CMat res(1, 1);

	for (size_t index = 0; index < gradients->size(); index++) {
		if (hiddenLayersSize != 0) {
			hiddenLayersSize--;
		}
		gradient = gradients->at(0);
		size_t weightIndexNum = weight->size() - 1 - index;
		if (index == gradients->size() - 1) {
			res.resize(gradient.getCol(), input->getCol());
			copyMat = input->getRowVal(mSampleRowNum);
			for (size_t r = 0; r < res.getRow(); r++) {
				for (size_t c = 0; c < res.getCol(); c++) {
					double value = gradient.getVal(0, r) * copyMat.getVal(0, c);
					if (std::fpclassify(value) != FP_SUBNORMAL) {
						res.changeVal(r, c, value);
					}
				}
			}
		}
		else {
			res.resize(gradient.getCol(), hiddenLayers->at(hiddenLayersSize)->getCol());
			copyMat = hiddenLayers->at(hiddenLayersSize)->getRowVal(mSampleRowNum);
			for (size_t r = 0; r < res.getRow(); r++) {
				for (size_t c = 0; c < res.getCol(); c++) {
					double value = gradient.getVal(0, r) * copyMat.getVal(0, c);
					if (std::fpclassify(value) != FP_SUBNORMAL) {
						res.changeVal(r, c, value);
					}
				}
			}
		}
		gradients->erase(gradients->begin());
		gradients->emplace_back(new CMat(&res));
	}
}
//test bias updeate
void CCalc::upadteBias(std::vector<CMat*>* biases, std::vector<CMat*>* gradients, std::vector<CMat*>* weight) {
 	CMat bias;
	CMat result(1,1);
	size_t biasIndexNum = 0;
	size_t row = 0;
	size_t col = 0;
	for (size_t index = 0; index < gradients->size(); index++) {
		biasIndexNum = biases->size() - 1 - index;
		row = biases->at(biasIndexNum)->getRow();
		col = biases->at(biasIndexNum)->getCol();
		bias = biases->at(biasIndexNum)->getRowVal(index);
		bias = bias - gradients->at(index);
		result.resize(row, col);
		for (size_t r = 0; r < row; r++) {
			for (size_t c = 0; c < col; c++) {
				result.changeVal(r, c, bias.getVal(0, c));
			}
		}

		*biases->at(biasIndexNum) = new CMat(&result);
	}
}

//가중치 업데이트
bool CCalc::updateWeight(CMat* input, std::vector<CMat*>* gradients, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers) {
	CMat gradient;
	size_t hiddenLayersSize = hiddenLayers->size();


	//gradients에 있는 gradient CMat*은 아래 과정 중 자연스럽게 메모리 해제가 이루어짐.
	upadteBias(biases, gradients, weight);
	fastMul(input, gradients, weight, biases, hiddenLayers);

	for (size_t index = 0; index < gradients->size(); index++) {
		if (hiddenLayersSize != 0) {
			hiddenLayersSize--;
		}
		gradient = gradients->at(index);

		size_t weightIndexNum = weight->size() - 1 - index;

	/*	for (size_t row = 0; row < gradient.getRow(); row++) {
			for (size_t col = 0; col < gradient.getCol(); col++) {
				std::cout << gradient.getVal(row, col) << " ";
			}
			std::cout << "\n";
		}*/

		gradient.multiplyByScalar(learningRate);
		gradient = gradient.transpose();
		*weight->at(weightIndexNum) = *weight->at(weightIndexNum) - &gradient;
	}

	return true;
}
//원핫인코딩
bool CCalc::oneHotEncoding(CMat* mat) {
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
double CCalc::calculateAccuracy(CMat* target, CMat* predictedValue) {
	double result = 0;
	double count = 0;
	CMat targetMat;
	CMat predicteMat;

	//int numArray[10] = { 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 };
	//int labelArray[10] = { 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 };

	if (target->getRow() != predictedValue->getRow() && target->getCol() != predictedValue->getCol()) {
		return result;
	}
	for (size_t r = 0; r < target->getRow(); r++) {
		targetMat = target->getRowVal(r);
		predicteMat = predictedValue->getRowVal(r);
		int labelNum = predictedNumber(&targetMat);
		int preadNum = predictedNumber(&predicteMat);
		//labelArray[labelNum]++;
		if (labelNum == preadNum) {
			//numArray[preadNum]++;
			count++;
		}
	}

	/*for (int i = 0; i < 10; i++) {
		std::cout << i << " : " << labelArray[i] << " / " << numArray[i] << " accuracy : " << (double)numArray[i] / (double)labelArray[i] << "\n";
	}*/

	result = count / target->getRow();
	return result;
}
//출력 matrix이랑 label matrix의 최대값(정답값 or 예측값) 찾아 int형태로 반환함
int CCalc::predictedNumber(CMat* mat) {
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
CMat* CCalc::forwardPropagation(CMat* inputLayer, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* weight, std::vector<CMat*>* biases, bool onOff) {
	//vector들의 size를 검사하여서 1번째 가중치 벡터와 입력을 relu하고 hidden 첫번째에 저장 그 다음엔 hidden 1과 weight2를 가지고 그 다음 hidden2나 return하기
	if (weight->size() <= 0) {
		return inputLayer;
	}
	delete hiddenLayers->at(0);
	hiddenLayers->erase(hiddenLayers->begin());
	hiddenLayers->emplace_back(selectActive(weight->at(0), inputLayer, biases->at(0)));
	/*if (onOff) {
		dropOut(hiddenLayers->at(hiddenLayers->size() - 1));
	}*/
	

	size_t nonChageValue = hiddenLayers->size() - 2;
	for (size_t index = 1; index < weight->size() - 1; index++) {
		delete hiddenLayers->at(0);
		hiddenLayers->erase(hiddenLayers->begin());
		hiddenLayers->emplace_back(selectActive(weight->at(index), hiddenLayers->at(nonChageValue), biases->at(index)));
		if (onOff) {
			dropOut(hiddenLayers->at(hiddenLayers->size() - 1));
		}
		//제일 앞을 지워야함.
		
	}

	if (weight->size() == 1) {
		return softmax(hiddenLayers->at(0));
	}

	return softmax(selectActive(weight->at(weight->size() - 1), hiddenLayers->at(hiddenLayers->size() - 1), biases->at(biases->size() - 1)));
}

//non drop out
CMat* CCalc::testForward(CMat* inputLayer, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* weight, std::vector<CMat*>* biases) {
	//vector들의 size를 검사하여서 1번째 가중치 벡터와 입력을 relu하고 hidden 첫번째에 저장 그 다음엔 hidden 1과 weight2를 가지고 그 다음 hidden2나 return하기
	if (weight->size() <= 0) {
		return inputLayer;
	}
	delete hiddenLayers->at(0);
	hiddenLayers->erase(hiddenLayers->begin());
	hiddenLayers->emplace_back(selectActive(weight->at(0), inputLayer, biases->at(0)));
	
	size_t nonChageValue = hiddenLayers->size() - 1;
	for (size_t index = 1; index < weight->size() - 1; index++) {
		//제일 앞을 지워야함.
		delete hiddenLayers->at(0);
		hiddenLayers->erase(hiddenLayers->begin());
		hiddenLayers->emplace_back(selectActive(weight->at(index), hiddenLayers->at(nonChageValue), biases->at(index)));
	}

	if (weight->size() == 1) {
		return softmax(hiddenLayers->at(0));
	}

	return softmax(selectActive(weight->at(weight->size() - 1), hiddenLayers->at(hiddenLayers->size() - 1), biases->at(biases->size() - 1)));
}
//Back Propagation
//랜덤으로 하나 뽑아서 그것으로만 진행하는 것으로 함. 미니배치 + 확률적 경사하강법 사용
bool CCalc::backwardPropagation(size_t repeatNum, CMat* input, CMat* output, CMat* target, std::vector<CMat*>* weight, std::vector<CMat*>* hiddenLayers, std::vector<CMat*>* biases) {
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
	isDone = gradinetDescent(input, &derivativeFunction, weight, biases, hiddenLayers, repeatNum);

	return isDone;
}