#include "cmatrix.h"
#include <cmath>
#include <random>
#include <iostream>
#include <vector>
#include <time.h>
#include <chrono>

const double minNum = 0.0;
const double maxNum = 0.02;

//constructors

CMat::CMat(size_t row, size_t col) {
    mRow = row;
    mCol = col;
    pVal = new double* [mRow];
    for (int r = 0; r < mRow; r++) {
        pVal[r] = new double[mCol];
        for (int c = 0; c < mCol; c++) {
            pVal[r][c] = 0;
        }
    }
}
CMat::CMat(CMat* mat) {
    mRow = mat->mRow;
    mCol = mat->mCol;
    pVal = new double* [mRow];
    for (int r = 0; r < mRow; r++) {
        pVal[r] = new double[mCol];
        for (int c = 0; c < mCol; c++) {
            pVal[r][c] = mat->getVal(r, c);
        }
    }
}
CMat::CMat(size_t row, size_t col, double val) {
    mRow = row;
    mCol = col;
    pVal = new double* [mRow];
    for (int r = 0; r < mRow; r++) {
        pVal[r] = new double[mCol];
        for (int c = 0; c < mCol; c++) {
            pVal[r][c] = val;
        }
    }
}

//fuctions

size_t CMat::getRow() {
    return this->mRow;
}
size_t CMat::getCol() {
    return this->mCol;
}
double** CMat::getModifiableVal() {
    return pVal;
}
double CMat::getVal(size_t row, size_t col) {
    return pVal[row][col];
}
void CMat::changeVal(size_t row, size_t col, double val) {
    pVal[row][col] = val;
}
CMat* CMat::getRowVal(size_t row) {
    CMat* result = new CMat(1, this->getCol());
    for (size_t c = 0; c < this->getCol(); c++) {
        result->changeVal(0, c, this->getVal(row, c));
    }

    return result;
}
CMat* CMat::getColVal(size_t col) {
    CMat* result = new CMat(this->getRow(), 1);
    for (size_t r = 0; r < this->getRow(); r++) {
        result->changeVal(r, 0, this->getVal(r, col));
    }

    return result;
}

//행렬 전치
CMat* CMat::transpose() {
    size_t row = mCol;
    size_t col = mRow;

    CMat* result = new CMat(row, col);

    for (int r = 0; r < row; r++) {
        for (int c = 0; c < col; c++) {
            result->changeVal(r, c, pVal[c][r]);
        }
    }

    return result;
}
//He초기화
CMat* CMat::replaceHeRandom() {
    size_t maxVal = 1;
    if (mRow > mCol) {
        maxVal = mRow;
    }
    else {
        maxVal = mCol;
    }
    //he 정규화
    double randomMax = std::sqrt(2.0 / maxVal);
    //double randomMax = std::sqrt(1.0 / (mRow + mCol));
    std::mt19937 gen;
    std::uniform_real_distribution<double> rDDouble(minNum, randomMax);

    //xavier 정규화
   /* double randomMax = std::sqrt(6.0 / (mRow + mCol));
    std::mt19937 gen;
    std::uniform_real_distribution<double> rDDouble(-randomMax, randomMax);*/

    for (int r = 0; r < mRow; r++) {
        for (int c = 0; c < mCol; c++) {
            double randomNumber = rDDouble(gen);
            pVal[r][c] = randomNumber;
        }
    }
    return this;
}
//스칼라 곱하기.
void CMat::multiplyByScalar(double scalar) {
    for (int r = 0; r < mRow; r++) {
        for (int c = 0; c < mCol; c++) {
            if (pVal[r][c] != 0) {
                pVal[r][c] *= scalar;//조금씩의 오차는 생김
            }
        }
    }
}
//unit matrix(단위 행렬)
void CMat::makeUnitMat() {
    if (mRow == mCol) {
        for (int r = 0; r < mRow; r++) {
            pVal[r][r] = 1;
        }
    }
}
//단위 행렬로 변환
CMat* CMat::convertUnitMat() {
    CMat copyMat(this);
    if (mRow == mCol) {
        this->makeUnitMat();
    }
    else if (mRow == 1 || mCol == 1) {
        size_t zVal = 0;
        mRow > mCol ? zVal = mRow : zVal = mCol;
        this->resize(zVal, zVal);
        bool isTrue = true;
        if (copyMat.getRow() == zVal) {
            isTrue = false;
        }
        for (size_t z = 0; z < zVal; z++) {
            if (isTrue) {
                pVal[z][z] = copyMat.getVal(0, z);
            }
            else {
                pVal[z][z] = copyMat.getVal(z, 0);
            }
        }
    }

    return this;
}
//Upper triangular matrix(상삼각행렬) optons이 0이면 순삼각행렬 default = 1
void CMat::makeUpperTriangularMat(int option) {
    CMat copyMat(this);
    if (mRow == mCol) {
        if (option != 0) {
            for (int r = 0; r < mRow; r++) {
                for (int c = r; c < mCol; c++) {
                    if (r != c) pVal[r][c] = copyMat.pVal[r][c];
                }
            }
        }
        else {
            for (int r = 0; r < mRow; r++) {//순삼각행렬
                for (int c = r; c < mCol; c++) {
                    pVal[r][c] = 0;
                }
            }
        }
    }
}
//Lower triangular matrix(하삼각행렬) optons이 0이면 순삼각행렬 default = 1
void CMat::makeLowerTriangularMat(int option) {
    CMat copyMat(this);
    if (mRow == mCol) {
        if (option != 0) {
            for (int r = 0; r < mRow; r++) {
                for (int c = 0; c < r; c++) {
                    if (r != c) pVal[r][c] = copyMat.pVal[r][c];
                }
            }
        }
        else {
            for (int r = 0; r < mRow; r++) {//순삼각행렬
                for (int c = 0; c < r; c++) {
                    pVal[r][c] = 0;
                }
            }
        }
    }
}

//trace (대각합)
double CMat::trace() {
    if (mRow != mCol) return 0;
    double result = 0;
    for (int r = 0; r < mRow; r++) {
        result += pVal[r][r];
    }
    return result;
}

//resize (0으로 초기화)
void CMat::resize(size_t row, size_t col) {
    for (int r = 0; r < mRow; r++) {
        delete[] pVal[r];
    }
    delete pVal;
    mRow = row;
    mCol = col;
    pVal = new double* [mRow];
    for (int r = 0; r < mRow; r++) {
        pVal[r] = new double[mCol];
        for (int c = 0; c < mCol; c++) {
            pVal[r][c] = 0;
        }
    }
}
//extend matrix (확장된 곳은 0으로 초기화)
void CMat::extendMat(size_t row, size_t col) {
    CMat copyMat(this);
    for (int r = 0; r < mRow; r++) {
        delete[] pVal[r];
    }
    delete pVal;
    mRow += row;
    mCol += col;
    pVal = new double* [mRow];
    for (int r = 0; r < mRow; r++) {
        pVal[r] = new double[mCol];
        for (int c = 0; c < mCol; c++) {
            if (r < copyMat.mRow && c < copyMat.mCol) {
                pVal[r][c] = copyMat.pVal[r][c];
            }
            else {
                pVal[r][c] = 0;
            }
        }
    }
}
bool CMat::pushBack(CMat* mat) {
    size_t pastRow = mRow;
    size_t pastCol = mCol;

    if (mat->getCol() < mCol && mat->getRow() < mRow) {
        return false;
    }

    if (mRow == 0 && mCol == 0) {
        resize(mat->getRow(), mat->getCol());
        for (size_t row = 0; row < mRow; row++) {
            for (size_t col = 0; col < mCol; col++) {
                pVal[row][col] = mat->getVal(row, col);
            }
        }
        return true;
    }

    if (mat->getCol() == mCol) {
        extendMat(mat->getRow(), 0);
        for (size_t row = pastRow; row < mRow; row++) {
            for (size_t col = 0; col < mCol; col++) {
                pVal[row][col] = mat->getVal(row - pastRow, col);
            }
        }
        return true;
    }

    if (mat->getRow() == mRow) {
        extendMat(0, mat->getCol());
        for (size_t row = 0; row < mRow; row++) {
            for (size_t col = pastCol; col < mCol; col++) {
                pVal[row][col] = mat->getVal(row, col - pastCol);
            }
        }
        return true;
    }

    return false;
}
//make symmetric matrix (대칭행렬), 1이면 위를 기준으로 대칭, 그 외는 아래를 기준으로 대칭 default = 1
void CMat::makeSymmetricMat(double val, int option) {
    //대각의 값은 살아 있어야하므로 순삼각행렬이 필요없음
    if (option == 1) {
        this->makeUpperTriangularMat();
    }
    else {
        this->makeLowerTriangularMat();
    }
    if (mRow == mCol) {
        for (int r = 0; r < mRow; r++) {
            for (int c = r; c < mCol; c++) {
                if (r != c) {
                    if (pVal[r][c] != 0) {
                        pVal[c][r] = pVal[r][c];
                    }
                    else {
                        pVal[r][c] = pVal[c][r];
                    }
                }
            }
        }
    }
}
//calculate determinant (가우스 소거법을 활용함)
double CMat::calculateDeterminant() {
    double result = 1;

    if (mRow != mCol) return 0;

    for (int r = 0; r < mRow; r++) {
        for (int i = r + 1; i < mCol; i++) {
            double valueOfMultiple = pVal[i][r] / pVal[r][r];//다음 행의 0이 아닌 첫 열을 0으로 만들기 위해 곱해야할 수를 찾기.
            for (int c = r; c < mCol; c++) {
                pVal[i][c] -= valueOfMultiple * pVal[r][c];
            }
        }
    }

    for (int i = 0; i < mRow; i++) {
        result *= pVal[i][i];
    }
    return result;
}
//normalize (0 ~ 1 사이의 값으로 변환)
void CMat::normalizeMat() {
    double minVal = 0;
    double maxVal = 1;

    for (int r = 0; r < mRow; r++) {
        for (int c = 0; c < mCol; c++) {
            if (minVal > pVal[r][c]) minVal = pVal[r][c];
            if (maxVal < pVal[r][c]) maxVal = pVal[r][c];
        }
    }
    for (int r = 0; r < mRow; r++) {
        for (int c = 0; c < mCol; c++) {
            pVal[r][c] = (pVal[r][c] - minVal) / (maxVal - minVal);
        }
    }
}

//operators
//A는 this, B는 operand
//덧셈
CMat* CMat::operator+(CMat* operand) {
    //만약 비어있다면 반환
    if (operand == nullptr) {
        return this;
    }
    //A와 B의 행렬이 모두 같아야 식이 성립 가능함
    if (mRow != operand->mRow || mCol != operand->mCol) {
        std::cout << "Please check and try again.\n";
        return nullptr;
    }
    //연산식
    CMat* result = new CMat(operand->mRow, operand->mCol);
    for (int r = 0; r < result->mRow; r++) {
        for (int c = 0; c < result->mCol; c++) {
            result->pVal[r][c] = pVal[r][c] + operand->pVal[r][c];
        }
    }
    return result;
}
//뺄셈
CMat* CMat::operator-(CMat* operand) {
    //만약 비어있다면 반환
    if (operand == nullptr) {
        return this;
    }
    //A와 B의 행렬이 모두 같아야 식이 성립 가능함
    if (mRow != operand->mRow || mCol != operand->mCol) {
        std::cout << "Please check and try again.\n";
        return nullptr;
    }
    //연산식;
    CMat* result = new CMat(operand->mRow, operand->mCol);
    for (int r = 0; r < result->mRow; r++) {
        for (int c = 0; c < result->mCol; c++) {
            result->pVal[r][c] = pVal[r][c] - operand->pVal[r][c];
        }
    }
    return result;
}
//곱셈
CMat* CMat::operator*(CMat* operand) {
    //만약 비어있다면 반환
    if (operand == nullptr) {
        return this;
    }
    size_t resultRow = 0;
    size_t resultCol = 0;
    //A = this, B = operand A와 B 중 A를 전치하면 1, B를 전치하면 0
    bool aOrBTranspose = 1;
    //A, B 둘 중 하나라도 transpose를 하면 1, 아니면 0
    bool isTranspose = 0;
    //A와 B의 순서를 바꾸어 계산 A를 전치해서 계산하여 다시 전치하는 것과 동일함
    bool aToBBToA = 0;

    size_t oRow = operand->mRow;
    size_t oCol = operand->mCol;
    CMat matTranspose(oCol, oRow);
    //변환을 해도 행렬의 곱셈연산이 불가능한 경우 this를 반환
    if (mRow != oRow && mRow != oCol && mCol != oRow && mCol != oCol) return this;
    //A와 B의 row와 col이 모두 같을 때에는 어떤 것도 건들지 않고 곱셈
    if (mRow == oRow && mCol == oCol && mRow == oCol) {
        resultRow = mRow;
        resultCol = mCol;
    }
    //A의 col과 B의 row가 같을 때 A의 row와 B의 col 크기의 행렬
    else if (mCol == oRow && mRow != oRow) {
        resultRow = mRow; 
        resultCol = oCol;
    }
    //A의 row와 B의 col이 같을 때 A의 col과 B의 row 크기의 행렬
    else if (mRow == oCol && mCol != oCol) {
        resultRow = oRow;
        resultCol = mCol;
        aToBBToA = 1;
    }
    //A와 B의 row가 같을 때 *A를 전치* A의 col과 B의 col 크기의 행렬
    else if (mRow == oRow) {
        resultRow = mCol;
        resultCol = oCol;
        matTranspose = this->transpose();
        isTranspose = 1;
    }
    //A와 B의 col이 같을 때 *B를 전치* A의 row와 B의 row 크기의 행렬
    else if (mCol == oCol) {
        resultRow = mRow;
        resultCol = oRow;
        matTranspose = operand->transpose();
        isTranspose = 1;
        aOrBTranspose = 0;
    }
    //혹시 모를 상황 대비
    else return this;

    CMat* result = new CMat(resultRow, resultCol);
    //연산식
    for (int r = 0; r < resultRow; r++) {
        for (int c = 0; c < resultCol; c++) {
            if (isTranspose) {
                if (aOrBTranspose) {
                    for (int newR = 0; newR < mRow; newR++) {//newR은 각각의 mat의 row값 단, transpose기준은 아님                
                        result->getModifiableVal()[r][c] += matTranspose.getVal(r, newR) * operand->getVal(newR, c);
                    }
                }
                else {
                    for (int newC = 0; newC < mCol; newC++) {//newC는 각각의 mat의 col값 단, transpose기준은 아님
                        result->getModifiableVal()[r][c] += this->getVal(r, newC) * operand->getVal(c, newC);
                    }
                }
            }
            else if (aToBBToA) {
                for (int newR = 0; newR < mRow; newR++) {//newR은 A : row, B : col을 나타냄.                    
                    result->getModifiableVal()[r][c] += operand->getVal(r, newR) * this->getVal(newR, c);
                }
            }
            else if (resultRow == mRow && resultCol == mCol) {
                for (int newR = 0; newR < oRow; newR++) {
                    result->getModifiableVal()[r][c] += this->getVal(r, c) * operand->getVal(newR, c);
                }
            }
            else {
                for (int newC = 0; newC < mCol; newC++) {//newC는 각각 A.Col, B.Row값을 나타낸다.                    
                    result->getModifiableVal()[r][c] += this->getVal(r, newC) * operand->getVal(newC, c);
                }
            }
        }
    }

    return result;
}
//비교
bool CMat::operator==(CMat* operand) {
    //만약 비어있다면 반환
    if (operand == nullptr) {
        return this;
    }
    //크기가 같지 않으면 바로 반환
    if (mRow != operand->mRow && mCol != operand->mCol) return false;
    //하나씩 비교하다가 하나라도 다르면 반환
    for (int r = 0; r < mRow; r++) {
        for (int c = 0; c < mCol; c++) {
            if (pVal[r][c] != operand->pVal[r][c]) return false;
        }
    }
    //위에 모든 과정을 통과하면 반환
    return true;
}
//대입
CMat* CMat::operator=(CMat* operand) {
    //만약 비어있다면 반환
    if (operand == nullptr) {
        return this;
    }
    //자신이 대입되었을 때 자신을 지우는 것을 방지
    if (this != operand) {
        for (int r = 0; r < mRow; r++) {
            delete[] pVal[r];
        }
        delete pVal;
    }
    //값 다시 지정
    mRow = operand->mRow;
    mCol = operand->mCol;
    pVal = new double* [mRow];
    //하나씩 대입
    for (int r = 0; r < mRow; r++) {
        pVal[r] = new double[mCol];
        for (int c = 0; c < mCol; c++) {
            pVal[r][c] = operand->pVal[r][c];
        }
    }
    //삭제를 통해 메모리 누수 방지
    delete operand;
    return this;
}
