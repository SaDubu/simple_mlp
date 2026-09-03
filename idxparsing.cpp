#include "idxpasing.h"
#include "cmatrix.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <stddef.h>

const uint32_t idx0Num = 0x00000800;

bool CParsing::parseIDXFile(std::string* filePath, CMat* mat, bool optionOfPrint) {
	clear();
	std::ifstream dataFile(*filePath, std::ios::binary);
	if (!dataFile.is_open()) {
		return false;
	}
	std::vector<uint8_t> valueDatas;
	bool normal = true;
	bool isMat = false;

	while (true) {
		dataFile.seekg(0);
		//��� ���� ���� �о���� �κ�
		//magic number 4byte �о����
		dataFile.read((char*)(&mMagicNumber), sizeof(mMagicNumber));
		mMagicNumber = convertEndian(mMagicNumber);
		//idx3 �����̸�
		if (mMagicNumber - idx0Num == 3) {
			//img ���Ϸ� �տ� magic number�� �����ϰ� 12byte�� �����
			//�̹����� �� ����
			dataFile.read((char*)(&mDataNumber), sizeof(mDataNumber));
			mDataNumber = convertEndian(mDataNumber);
			//�� ��
			dataFile.read((char*)(&mPixelRow), sizeof(mPixelRow));
			mPixelRow = convertEndian(mPixelRow);
			//�� ��
			dataFile.read((char*)(&mPixelCol), sizeof(mPixelCol));
			mPixelCol = convertEndian(mPixelCol);
		}
		//idx1�����̸�
		else if (mMagicNumber - idx0Num == 1) {
			//label ���Ϸ� �տ� magic number�� �����ϰ� 4byte�� �����
			//�̹����� �� ����
			dataFile.read((char*)(&mDataNumber), sizeof(mDataNumber));
			mDataNumber = convertEndian(mDataNumber);
		}
		else if (mMagicNumber - idx0Num == 2) {
			dataFile.close();
			do {
				isMat = readMat(*filePath, mat);
			} while (!isMat);
			break;
		}
		else {
			normal = false;
			break;
		}

		//������� �о�� ������ �������� data���� vector resizing
		valueDatas.resize((mDataNumber * (mPixelRow * mPixelCol)));
		//�����͸� �� ���� �� ���� �ֱ�
		dataFile.read((char*)valueDatas.data(), valueDatas.size());
		size_t readEnd = dataFile.tellg();
		dataFile.seekg(0, std::ios::end);
		size_t seekgEnd = dataFile.tellg();
		//file_size�� �Ķ���ʹ� ������ ��ο����� �׷��� ���������ͷ� �޾ƿ��� �ʰ� ��η� �޾ƿ�
		if (readEnd == seekgEnd) {
			normal = true;
			break;
		}
	}
	//���� �бⰡ �������ϱ� �ݱ�
	dataFile.close();

	if (!normal) {
		return false;
	}
	if (isMat) {
		return true;
	}
	mDataShape->resize(mDataNumber, (mPixelRow * mPixelCol));
	size_t row = 0;
	for (size_t valueDatasIndex = 0; valueDatasIndex < valueDatas.size(); valueDatasIndex++) {
		if (valueDatasIndex != 0) {
			if (valueDatasIndex % (mPixelRow * mPixelCol) == 0) {
				row++;
			}
		}
		mDataShape->changeVal(row, valueDatasIndex % (mPixelRow * mPixelCol), valueDatas[valueDatasIndex]);
	}
	
	CMat* copyMat = new CMat(mDataShape);

	*mat = copyMat;

	clear();

	return true;
}

bool CParsing::DifferentiateEndian(int value) {
	//little-endian�� �ڿ������� ������ �ϱ� ������ �Ʒ��� ���� 0x00000001�� ���� 0x01000000�� ������ �� (��ǥ���� ��  Intel)
	//big-endian�� 0x00000001�� ���� 0x00000001 �״�� ������ �� (��ǥ���� ��  Sparc) <- ��Ʈ��ũ ȯ�濡���� �̰����� ����
	if (((char*)&value)[0]) //little-endian
		return false;
	else // big-endian;
		return true;
}

int CParsing::convertEndian(int value) {
	unsigned char bytes[4];
	int result = 0;
	if (DifferentiateEndian(value)) { // big -> little1 64 
		bytes[0] = (unsigned char)((value >> 0) & 255);
		bytes[1] = (unsigned char)((value >> 8) & 255);
		bytes[2] = (unsigned char)((value >> 16) & 255);
		bytes[3] = (unsigned char)((value >> 24) & 255);

		result =
			((int)bytes[0] << 24) +
			((int)bytes[1] << 16) +
			((int)bytes[2] << 8) +
			((int)bytes[3] << 0);
		return result;
	}

	return value;
}

bool CParsing::saveMat(CMat* mat, std::string* filePath) {
	std::ofstream saveFile(*filePath, std::ios::binary);
	uint32_t idx2Num = 0x00000802;
	if (!saveFile.is_open()) {
		return false;
	}
	size_t row = mat->getRow();
	size_t col = mat->getCol();

	saveFile.write((char*)(&idx2Num), sizeof(idx2Num));

	saveFile.write((char*)(&row), sizeof(row));

	saveFile.write((char*)(&col), sizeof(col));

	std::vector<double> saveData;

	for (size_t r = 0; r < row; r++) {
		for (size_t c = 0; c < col; c++) {
			double saveValue = mat->getVal(r, c);
			saveData.emplace_back(saveValue);
		}
	}
	saveFile.write((char*)saveData.data(), saveData.size() * sizeof(double));

	saveFile.close();
	return true;
}

bool CParsing::readMat(std::string filePath, CMat* mat) {
	std::vector<double> data;

	std::ifstream savedMat(filePath, std::ios::binary);
	if (!savedMat.is_open()) {
		return false;
	}

	size_t row = 0;
	size_t col = 0;

	savedMat.seekg(sizeof(mMagicNumber));
	//�� ��
	savedMat.read((char*)(&row), sizeof(row));
	//�� ��
	savedMat.read((char*)(&col), sizeof(col));

	data.resize(row * col);

	savedMat.read((char*)data.data(), data.size()*sizeof(double));

	size_t readEnd = savedMat.tellg();
	savedMat.seekg(0, std::ios::end);
	size_t seekgEnd = savedMat.tellg();

	savedMat.close();

	if (readEnd != seekgEnd) {
		return false;
	}

	mDataShape->resize(row, col);

	size_t indexNumber = 0;
	for (size_t r = 0; r < row; r++) {
		for (size_t c = 0; c < col; c++) {
			mDataShape->changeVal(r, c, data[indexNumber]);
			indexNumber++;
		}
	}

	CMat* copyMat = new CMat(mDataShape);

	*mat = copyMat;

	clear();

	return true;
}

void CParsing::saveTxt(std::vector<std::string>* data, std::string* filePath) {
	std::ofstream txtFile(*filePath, std::ios::app);

	for (std::string str : *data) {
		txtFile << str << " ";
	}

	txtFile << "\n";

	txtFile.close();
}

void CParsing::clear() {
	mMagicNumber = 1;
	mDataNumber = 1;
	mPixelRow = 1;
	mPixelCol = 1;
	mDataShape->resize(0, 0);
}