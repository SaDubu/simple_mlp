#pragma once
//내가 사용하고자 하는 만큼의 스레드를 사용할 수 있는지 판단

#include <thread>

class CThreadCheck {
private :
	int threadNum = std::thread::hardware_concurrency();
public:
	inline bool checkNum(int num) {
		return threadNum > num;
	}

	~CThreadCheck() {};
};