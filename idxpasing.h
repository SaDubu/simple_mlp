#pragma once

#include "cmatrix.h"

#include <string>
#include <stddef.h>
#include <vector>

class CParsing {
private:
	std::string mFilePath = "";
	uint32_t mMagicNumber;
	uint32_t mDataNumber;
	uint32_t mPixelRow;
	uint32_t mPixelCol;
	CMat* mDataShape;
public:
	CParsing() {
		mMagicNumber = 1;
		mDataNumber = 1;
		mPixelRow = 1;
		mPixelCol = 1;
		mDataShape = new CMat(1, 1); 
	};

	bool parseIDXFile(std::string* filePath, CMat* mat, bool optionOfPrint = false);

	bool DifferentiateEndian(int value);

	int convertEndian(int value);

	bool saveMat(CMat* mat, std::string* filePath);

	bool readMat(std::string filePath, CMat* mat);

	void saveTxt(std::vector<std::string>* data, std::string* filePath);

	void clear();

	~CParsing() {
		delete mDataShape;
	};
};