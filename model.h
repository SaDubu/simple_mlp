#pragma once

#include "cmatrix.h"

#include <vector>
#include <string>
#include <stddef.h>

class CModel {
private:
	int threadNum = 4;

	std::vector<CMat*> mHiddenLayers;
	std::vector<CMat*> mWeights;
	std::vector<CMat*> mBiases;
	std::vector<CMat*> mBatch;
	std::vector<CMat*> mTargetBatch;

	std::vector<CMat*> mTestBatch;
	std::vector<CMat*> mTestTargetBatch;

	std::vector<std::vector<CMat*>> mSplitBatch;

	size_t mHiddenLayerSize = 0;
	size_t mHiddenLayerNumber = 0;
	size_t mBatchSize = 0;

	CMat* mTestInput;
	CMat* mTestLabel;

	CMat* mInputLayer;
	CMat* mOutputLayer;
	CMat* mTargetMat;
public:
	CModel() {
		mHiddenLayers.resize(0);
		mWeights.resize(0);
		mBiases.resize(0);
		mBatch.resize(0);
		mTargetBatch.resize(0);

		mTestBatch.resize(0);
		mTestTargetBatch.resize(0);

		mSplitBatch.resize(0);

		mTestInput = new CMat(1, 1);
		mTestLabel = new CMat(1, 1);

		mInputLayer = new CMat(1, 1);
		mOutputLayer = new CMat(1, 1);
		mTargetMat = new CMat(1, 1);
	};

	void setS(std::string a);

	bool setHiddengLayerNum();

	bool makeHidenLayer();

	bool makeWeight();

	bool replaceWeight();

	bool makeModel(const std::string* imgFilePath,
		const std::string* targetFilePath,
		bool isLoad = false,
		bool isMiniBatch = false);

	bool saveModel();

	bool loadModel();

	bool makeMiniBatch();

	bool makeTestBatch();

	void clearLayer();

	bool batchLearning(size_t epoch);

	bool splitOfBatch();

	bool threadBatchLearning(size_t epoch);

	bool traingUseThread();

	bool traing();

	void runAll(size_t saveBatchSize, int epoch);

	bool test(size_t saveBatchSize);

	bool mtest(size_t saveBatchSize);

	bool setTestData(const std::string* imgFilePath,
		const std::string* targetFilePath);

	bool propgate(double* loss, CMat* propagetResult, size_t* epoch, int indexNum);

	bool saveBestModel();

	void resetBias();

	inline size_t getBatchSize() {
		return mBatchSize;
	}

	inline double calculateTotalLoss(double loss) {
		return loss / (mInputLayer->getRow() / mBatchSize);
	}

	inline int getUsingThreadNum() {
		return threadNum;
	}
	
	~CModel() {
		delete mInputLayer;
		delete mOutputLayer;
		delete mTargetMat;
		delete mTestInput;
		delete mTestLabel;

		for (int index = 0; index < mHiddenLayers.size(); index++) {
			delete mHiddenLayers.at(index);
		}

		for (int index = 0; index < mWeights.size(); index++) {
			delete mWeights.at(index);
		}

		for (int index = 0; index < mBiases.size(); index++) {
			delete mBiases.at(index);
		}

		for (int index = 0; index < mBatch.size(); index++) {
			delete mBatch.at(index);
		}

		for (int index = 0; index < mTargetBatch.size(); index++) {
			delete mTargetBatch.at(index);
		}

		for (int index = 0; index < mTestBatch.size(); index++) {
			delete mTestBatch.at(index);
		}

		for (int index = 0; index < mTestTargetBatch.size(); index++) {
			delete mTestTargetBatch.at(index);
		}

		for (int x = 0; x < mSplitBatch.size(); x++) {
			for (int y = 0; y < mSplitBatch[x].size(); y++) {
				delete mSplitBatch.at(x).at(y);
			}
		}
		mHiddenLayers.clear();
		mWeights.clear();
		mBiases.clear();
		mBatch.clear();
		mTargetBatch.clear();
		mSplitBatch.clear();
		std::vector<CMat*>().swap(mHiddenLayers);
		std::vector<CMat*>().swap(mWeights);
		std::vector<CMat*>().swap(mBiases);
		std::vector<CMat*>().swap(mBatch);
		std::vector<CMat*>().swap(mTargetBatch);
		std::vector<CMat*>().swap(mTestBatch);
		std::vector<CMat*>().swap(mTestTargetBatch);
		std::vector<std::vector<CMat*>>().swap(mSplitBatch);
	}
};