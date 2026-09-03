#pragma once

#include <stddef.h>

class CMat {
private:
    size_t mRow;
    size_t mCol;
    double** pVal;
public:
    CMat() { mRow = 0; mCol = 0; pVal = nullptr; };
    CMat(size_t row, size_t col);
    CMat(CMat* mat);
    CMat(size_t row, size_t col, double val);

    size_t getRow();
    size_t getCol();
    double** getModifiableVal();
    double getVal(size_t row, size_t col);
    void changeVal(size_t row, size_t col, double val);
    CMat* getRowVal(size_t row);
    CMat* getColVal(size_t col);

    CMat* transpose();

    CMat* replaceHeRandom();

    CMat* convertUnitMat();

    void multiplyByScalar(double scalar);
    void makeUnitMat();
    void makeLowerTriangularMat(int option = 1);
    void makeUpperTriangularMat(int option = 1);
    void resize(size_t row, size_t col);
    void extendMat(size_t row, size_t col);
    bool pushBack(CMat* mat);
    void makeSymmetricMat(double val, int option = 1);
    double calculateDeterminant();
    void normalizeMat();

    double trace();

    CMat* operator+(CMat* operand);
    CMat* operator-(CMat* operand);
    CMat* operator*(CMat* operand);
    bool operator==(CMat* operand);
    CMat* operator=(CMat* operand);

    //destructor
    ~CMat() {
        for (int row = 0; row < mRow; row++) {
            delete[] pVal[row];
        }
        delete pVal;
    }
};