#pragma once

#include <string>

class CIo {
private:
	int mInt;
	double mDouble;
	std::string mString;
	char mChar;
public:
	CIo() { mInt = 0; mDouble = 0.0; mString = ""; mChar = NULL; };

	bool inputInt(int* value);
	bool inputDouble(double* value);
	bool inputString(std::string* value);
	bool inputChar(char* value);

	bool output(const std::string* text);
	bool printLearnStat(size_t epoch, double totalLoss, double accuracy, size_t inputLayerRow);

	~CIo() {};
};