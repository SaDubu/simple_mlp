#include "executing.h"
// //메모리 누수
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>


#ifdef _DEBUG
#define new new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#endif
//------

//
// 2024.06.13 스레드 미니배치 병렬처리, 미니배치, online 가능
// 전체적으로 아직 정리가 다 되지 않은 느낌이라 보다 더 깔끔하게 만들 것.
//


int main() {
	CExecute exe;

	exe.actModel();

	return 0;
}

//register int  : register에게 잡아놓으라고 하는 것으로 빠름
//volatile int  : 컴파일러의 최적화를 막음.



//#include <vector>
//#include "cmatrix.h"
//#include "calcmat.h"
//#include <cmath>
//#include <math.h>
//#include <time.h>
//#include <iostream>
//#include <chrono>
//
//size_t mSampleRowNum = 0;
//double learningRate = 0.01;
//
//#define CHECK_DURATION		1
//
//
//bool updateWeight(CMat* input, std::vector<CMat*>* gradients, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers) {
//	CMat gradient;
//	size_t hiddenLayersSize = hiddenLayers->size();
//
//	//gradients에 있는 gradient CMat*은 아래 과정 중 자연스럽게 메모리 해제가 이루어짐.
//
//#if CHECK_DURATION
//	int64_t tookNanos = 0;
//	std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<__int64, std::ratio<1, 1000000000000>>> start, end;
//#endif
//
//	for (size_t index = 0; index < gradients->size(); index++) {
//		if (hiddenLayersSize != 0) {
//			--hiddenLayersSize;
//		}
//		gradient = gradients->at(index);
//		CMat copyMat;
//		CMat res(1, 1);
//		size_t weightIndexNum = weight->size() - 1 - index;
//
//		if (index < gradients->size()) {
//
//			res.resize(gradient.getCol(), hiddenLayers->at(hiddenLayersSize)->getCol());
//			copyMat = hiddenLayers->at(hiddenLayersSize)->getRowVal(mSampleRowNum);
//			for (size_t r = 0; r < res.getRow(); r++) {
//				for (size_t c = 0; c < res.getCol(); c++) {
//					double value = gradient.getVal(0, r) * copyMat.getVal(0, c);
//
//#if CHECK_DURATION
//					start = std::chrono::high_resolution_clock::now();
//#endif
//
//					if (std::fpclassify(value) == FP_NORMAL) {
//
//#if CHECK_DURATION
//						end = std::chrono::high_resolution_clock::now();
//						tookNanos = (end - start).count();
//
//						std::cout << "Elapsed: " << tookNanos << " ns" << std::endl;
//#endif
//						
//						res.changeVal(r, c, value);
//						continue;
//					}
//				}
//			}
//		}
//		else {
//			start = std::chrono::high_resolution_clock::now();
//			res.resize(gradient.getCol(), input->getCol());
//			copyMat = input->getRowVal(mSampleRowNum);
//			for (size_t r = 0; r < res.getRow(); r++) {
//				for (size_t c = 0; c < res.getCol(); c++) {
//					double value = gradient.getVal(0, r) * copyMat.getVal(0, c);
//
//					{
//
//						res.changeVal(r, c, value);
//					}
//				}
//			}
//			end = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double, std::milli> elapsed = end - start;
//
//			//std::cout << "Elapsed time: " << elapsed.count() * 1000 << " 마이크로" << std::endl;
//
//			start = std::chrono::high_resolution_clock::now();
//			CMat r = copyMat * &gradient;
//			r = r.transpose();
//			end = std::chrono::high_resolution_clock::now();
//			elapsed = end - start;
//
//			//std::cout << "Elapsed time: " << elapsed.count() * 1000 << " 마이크로" << std::endl;
//			if (r == &res) {
//				r.resize(0, 0);
//			}
//		}
//
//		if (index == gradients->size() - 1) {
//			start = std::chrono::high_resolution_clock::now();
//			res.resize(gradient.getCol(), input->getCol());
//			copyMat = input->getRowVal(mSampleRowNum);
//			for (size_t r = 0; r < res.getRow(); r++) {
//				for (size_t c = 0; c < res.getCol(); c++) {
//					double value = gradient.getVal(0, r) * copyMat.getVal(0, c);
//					
//					 {
//						
//						res.changeVal(r, c, value);
//					}
//				}
//			}
//			end = std::chrono::high_resolution_clock::now();
//			std::chrono::duration<double, std::milli> elapsed = end - start;
//
//			//std::cout << "Elapsed time: " << elapsed.count() * 1000 << " 마이크로" << std::endl;
//			
//			start = std::chrono::high_resolution_clock::now();
//			CMat r = copyMat * &gradient;
//			r = r.transpose();
//			end = std::chrono::high_resolution_clock::now();
//			elapsed = end - start;
//
//			//std::cout << "Elapsed time: " << elapsed.count() * 1000 << " 마이크로" << std::endl;
//			if (r == &res) {
//				r.resize(0, 0);
//			}
//
//		}
//		else {
//			res.resize(gradient.getCol(), hiddenLayers->at(hiddenLayersSize)->getCol());
//			copyMat = hiddenLayers->at(hiddenLayersSize)->getRowVal(mSampleRowNum);
//			for (size_t r = 0; r < res.getRow(); r++) {
//				for (size_t c = 0; c < res.getCol(); c++) {
//					double value = gradient.getVal(0, r) * copyMat.getVal(0, c);
//					start = std::chrono::high_resolution_clock::now();
//
//					if (std::fpclassify(value) == FP_NORMAL) {
//
//					}
//					if (std::fpclassify(value) != FP_SUBNORMAL) {
//						end = std::chrono::high_resolution_clock::now();
//						std::chrono::duration<double, std::nano>elapsed = end - start;
//
//						std::cout << "Elapsed time: " << elapsed.count() << " ns" << std::endl;
//						res.changeVal(r, c, value);
//					}
//				}
//			}
//		}
//		//값을 확인하는 것을 하고 안하고의 차이는 64 10 기준 평균 0.02ms 784 64 기준 평균 0.04ms가량 차이가 남.
//		//값을 확인하는 것에 걸리는 시간은 64 10 기준 평균 0.33ms 784 64 가쥰 평균 0.6ms가 걸린다.
//		//또 값을 확인하지 않고 아래 루프를 도는 것과 연산자 오버로딩을 이용하는 것은
//		//또 changeVal 자체는 64 10 기준 평균 0.05ms 784 64 기준 0.12ms가 걸림.
//		//double value = weight->at(weightIndexNum)->getVal(r, c) - res.getVal(c, r); <- 이 부분
//		//64 10 기준 평균 0.048ms 784 64 기준 평균 0.11ms가 걸림.
//		//그래서 총 연산은 64 10 기준 평균 0.098ms 784 64 기준 평균 0.24ms가 걸림 
//		//루프를 타지 않았을 때 64 10 기준 평균 0.04ms 784 64 기준 평균 0.1ms
//		// 소숫점 4번째 자리에서 반올림 
//		//루프 탄 경우 64 10 총 63ms 타지 않은 경우 64 10 총 28ms
//		//루프 탄 경우 784 64 총 11859ms 타지 않은 경우 784 64 총 4992ms
//		// 루프 탄 경우 총 11921ms 타지 않은 경우 5020ms
//		// 모든 계산은 소숫점 소숫점 자리 반올림함. 1번째자리
//		//위와 같이 소요됨.
//		// 디버깅 마다 계속 소요 값은 계속 변하긴 하지만 위의 보이는 것처럼 루프를 탄 경우와 타지 않은 경우의 값의 차이는 벌어져 있음.
//
//		//연산 시간은 소숫점 자리 때문에도 시간 차이가 벌어짐. 아래와 같이 값이 정해져있지 않은 실제 학습인 경우엔 소숫점 n자리 n이 클수록 연산 시간은 늘어남.
//
//
//		CMat w(weight->at(weightIndexNum));
//		res.multiplyByScalar(learningRate);
//		
//		int64_t a = 0, b = 0;
//		start = std::chrono::high_resolution_clock::now();
//		for (size_t r = 0; r < weight->at(weightIndexNum)->getRow(); r++) {
//			for (size_t c = 0; c < weight->at(weightIndexNum)->getCol(); c++) {
//				double value = weight->at(weightIndexNum)->getVal(r, c) - res.getVal(c, r);
//
//				//std::cout << "Elapsed: " << tookNanos << " ns" << std::endl;
//				
//				if (!std::isnan(value) || std::fpclassify(value) != FP_SUBNORMAL || !std::isinf(value)) {
//					weight->at(weightIndexNum)->changeVal(r, c, value);
//					
//				}
//				
//				/*if (r == biases->at(weightIndexNum)->getRow() - 1) {
//					double valueOfBias = biases->at(weightIndexNum)->getVal(0, c) - gradient.getVal(0, c);
//					
//					if (!std::isnan(valueOfBias) || std::fpclassify(valueOfBias) != FP_SUBNORMAL || !std::isinf(valueOfBias)) {
//
//						for (size_t biasRow = 0; biasRow < biases->at(weightIndexNum)->getRow(); biasRow++) {
//							biases->at(weightIndexNum)->changeVal(biasRow, c, valueOfBias);
//						}
//					}
//				}*/
//			}
//		}
//		end = std::chrono::high_resolution_clock::now();
//		tookNanos = (end - start).count();
//		a += tookNanos;
//		
//		start = std::chrono::high_resolution_clock::now();
//		res = res.transpose();
//		CMat r = w - &res;
//		end = std::chrono::high_resolution_clock::now();
//		tookNanos = (end - start).count();
//		b += tookNanos;
//		std::cout << "reuslut " << tookNanos << "\n";
//		std::wcout << a << " " << b << "\n";
//		if (r == &w) {
//			r.resize(0, 0);
//		}
//	}
//
//	return true;
//}
//
//
//
////
//// 시험 해볼 것
//// 1. forward, backPropagation, calculate loss 에 걸리는 시간을 각각 측정해볼 것
//// 2. 위 각각의 연산에 vector[index]형태로 들어가는 인자를 그냥 CMat에 담아 넣어서 시간 측정해보고 메모리 관련도 확인 해볼 것
//// 3. 시간이 오래 걸리는 곳을 찾았다면 그 곳을 어떻게 줄일 수 있는지 관련 연산 부분을 가져와서 제대로 확인해볼 것.
////
//
////
//// 시험하기 위해 필요한 것 실제 데이터를 가지고 진행을 하는 것이 좋을지, 아니면 내가 만든 임의 데이터로 하는게 좋을지 고민해봐야함 실제 데이터로 한다면 실제로 걸리는 시간을 측정할 수 있음.
//// 하지만 내가 만든 임의 데이터로 한다면 실제 걸리는 시간보다 더 걸릴 가능성이 있어보임 사유 : 부동 소수점 수가 내가 임의로 만든 것이 훨씬 크게 되어 있어 연산 시간에 좀 더 소요가 될 것 같음.
//// 자세한 사항은 측정을 해보아야하는 것이지만 임의 데이터로 한다고 한들 실질적으로 연산 별 시간 차이를 보기 위해 일관된 데이터로 한다면 어떤 데이터든 상관하지 않고 일정한 비율이 될 것이라 생각됨.
//// 따라서 임의 데이터로 실험할 에정. 만약 결과가 내 예상과 다르게 나온다면 실제 연산에 쓰이는 데이터를 가져와서 진행할 예정임.
////
//
////
//// 시험 용 변수들 선언부
////
//std::vector<CMat*> mBatch, mTargetBatch, mHiddenLayers, mWeights, mBiases;
//CMat* mOutputLayer;
//
//bool batchLearning(size_t epoch) {
//	CCalc calcMat;
//	calcMat.setActName("elu");
//	calcMat.setOptimizer("sgd");
//	double loss = 0;
//	double tLoss = 0;
//	double totalLoss = 0;
//	double accuracy = 0;
//	int miniEpoch = 0;
//	int maxME = 0;
//
//	bool isDone = true;
//
//	CMat outputPerEpoch(0, 0);
//
//	int64_t tookNanos = 0;
//	std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<__int64, std::ratio<1, 1000000000000>>> start, end;
//	start = std::chrono::high_resolution_clock::now();
//	for (int index = 0; index < mBatch.size(); index++) {
//		
//		*mOutputLayer = calcMat.forwardPropagation(mBatch[index],
//			&mHiddenLayers, &mWeights, &mBiases, true);
//		
//
//		//loss += calcMat.meanSquareError(mTargetBatch[index], mOutpu tLayer);
//		loss = calcMat.categoricalCrossEntropy(mTargetBatch[index], mOutputLayer);
//
//		isDone = calcMat.backwardPropagation(epoch + 1,
//			mBatch[index], mOutputLayer, mTargetBatch[index],
//			&mWeights, &mHiddenLayers, &mBiases);
//
//		if (!isDone) {
//			break;
//		}
//
//		if ((loss < 0.0) || miniEpoch != 0) {
//			/*isDone = outputPerEpoch.pushBack(mOutputLayer);
//			if (!isDone) {
//				return false;
//			}*/
//
//			tLoss += loss;
//
//			if (maxME < miniEpoch) {
//				maxME = miniEpoch;
//			}
//			miniEpoch = 0;
//		}
//		else {
//			miniEpoch++;
//			index--;
//		}
//
//		/*isDone = outputPerEpoch.pushBack(mOutputLayer);
//		if (!isDone) {
//			return false;
//		}*/
//
//
//	}
//	end = std::chrono::high_resolution_clock::now();
//	tookNanos = (end - start).count();
//
//	if (!isDone) {
//		return false;
//	}
//
//	//totalLoss = calculateTotalLoss(tLoss);
//	/*accuracy = calcMat.calculateAccuracy(mTargetMat, &outputPerEpoch);*/
//
//	std::cout << "max mini epoch is " << maxME;
//
//	//std::cout << "\ntraining : ";
//	//isDone = inOut.printLearnStat(epoch, totalLoss, 1, 1);
//
//	return isDone;
//}
//
//
//
//int main() {
//	int64_t tookNanos = 0;
//	std::chrono::time_point<std::chrono::steady_clock, std::chrono::duration<__int64, std::ratio<1, 1000000000000>>> start, end;
//	//CMat* input, std::vector<CMat*>* gradients, std::vector<CMat*>* weight, std::vector<CMat*>* biases, std::vector<CMat*>* hiddenLayers
//	std::vector<CMat*> a, b, c, d;
//
//	CMat in(10000, 784, 0.012312);
//
//	CMat hidden1(10000, 64, 0.1231);
//
//	CMat weight1(784, 64, 0.213121);
//	CMat weight2(64, 10, 0.002301);
//
//	CMat bias1(10000, 64, 0.00321401);
//	CMat bias2(10000, 10, 0.02131401);
//
//	CMat g1(1, 10, 0.001);
//	CMat g2(1, 64, 0.0001);
//
//	CMat target(10000, 10);
//	CMat rfs(0, 0);
//
//	mOutputLayer = new CMat(1, 1);
//
//	for (size_t r = 0; r < target.getRow(); r++) {
//		target.getModifiableVal()[r][0] = 1.0;
//	}
//
//	
//	
//
//	mBatch.emplace_back(new CMat(&in));
//	mTargetBatch.emplace_back(new CMat(&target));
//
//	mHiddenLayers.emplace_back(new CMat(&hidden1));
//
//	mWeights.emplace_back(new CMat(&weight1));
//	mWeights.emplace_back(new CMat(&weight2));
//
//	mBiases.emplace_back(new CMat(&bias1));
//	mBiases.emplace_back(new CMat(&bias2));
//
//	batchLearning(0);
//
//	a.emplace_back(new CMat(&g1));
//	a.emplace_back(new CMat(&g2));
//
//	b.emplace_back(new CMat(&weight1));
//	b.emplace_back(new CMat(&weight2));
//
//	c.emplace_back(new CMat(&bias1));
//	c.emplace_back(new CMat(&bias2));
//
//	d.emplace_back(new CMat(&hidden1));
//
//	updateWeight(&in, &a, &b, &c, &d);
//
//	return 0;
//}