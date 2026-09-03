#pragma once

#include <string>
#include <stddef.h>

class CExecute {
private :
	const std::string testImgPath = "mnist_dataset/t10k-images.idx3-ubyte";
	const std::string testLabelPath = "mnist_dataset/t10k-labels.idx1-ubyte";

	const std::string trianImgPath = "mnist_dataset/train-images.idx3-ubyte";
	const std::string trianLabelPath = "mnist_dataset/train-labels.idx1-ubyte";

	const std::string saveModelPath = "saveModel";

	const std::string saveBestPath = "Best_Accuary";
public :
	bool actModel();

	bool proceed();

	~CExecute() {};
};