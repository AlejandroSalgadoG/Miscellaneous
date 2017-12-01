#include <iostream>
#include <fstream>
#include <sstream>

#include "../Debug/Functions.h"

using namespace std;

int main(int argc, char *argv[]) {

	if (argc != 4) {
		cerr << "Help: " << endl;
		cerr << "\tPractica2 <matrix A> <matrix B> <operation>\n" << endl;
		cerr << "\t<matrix A> = Path to the file that contains the matrix A" << endl;
		cerr << "\t<matrix B> = Path to the file that contains the matrix B" << endl;
		cerr << "\t<operation> = + | - | * | /" << endl;
		return -1;
	}

	ifstream matrixA(argv[1]);
	ifstream matrixB(argv[2]);

	if (!matrixA.good() || !matrixB.good()) {
		cerr << "files not founded" << endl;
		return -1;
	}

	float A[16]; string lineA;
	float B[16]; string lineB;
	float C[16];

	for (int i = 0; i<4; i++) {
		getline(matrixA, lineA);
		getline(matrixB, lineB);

		istringstream inA(lineA);
		istringstream inB(lineB);

		inA >> A[i * 4] >> A[i * 4 + 1] >> A[i * 4 + 2] >> A[i * 4 + 3];
		inB >> B[i * 4] >> B[i * 4 + 1] >> B[i * 4 + 2] >> B[i * 4 + 3];
	}

	matrixA.close();
	matrixB.close();

	char operation = *argv[3];

	switch (operation) {
		case '+': add(A, B, C); break;
		case '-': sub(A,B,C);	break;
		case '*': mul(A, B, C);  break;
		case '/': div(A, B, C);  break;
		default:
			cout << "Operation not recongized" << endl;
			return -1;
	}

	for (int i = 0; i<4; i++) {
		for (int j = 0; j<4; j++) cout << A[i * 4 + j] << " "; cout << "\t";
		for (int j = 0; j<4; j++) cout << B[i * 4 + j] << " "; cout << "\t";
		for (int j = 0; j<4; j++) cout << C[i * 4 + j] << " "; cout << "" << endl;
	}

	return 0;

}