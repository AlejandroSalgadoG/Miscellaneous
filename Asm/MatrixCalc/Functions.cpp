#include "Functions.h"

void add(float * A, float * B, float * C) {
	_asm {
		mov ebx, B[0]
		mov eax, [ebx]
		push eax
		mov ebx, A[0]
		mov eax, [ebx]
		push eax
		call pSuma
		fstp C[0]
		pop eax
		pop eax

		mov ebx, 15

		add1:
			mov eax, B[4 * ebx]
			push eax
			mov eax, A[4 * ebx]
			push eax
			call pSuma
			fstp C[4 * ebx]
			pop eax
			pop eax

			dec ebx
			jnz add1
	}
}

void sub(float * A, float * B, float * C) {
	_asm {
		mov eax, B[0]
		push eax
		mov eax, A[0]
		push eax
		call pResta
		fstp C[0]
		pop eax
		pop eax

		mov ebx, 15

		sub1:
			mov eax, B[4 * ebx]
			push eax
			mov eax, A[4 * ebx]
			push eax
			call pResta
			fstp C[4 * ebx]
			pop eax
			pop eax

			dec ebx
			jnz sub1
	}
}

void mul(float * A, float * B, float * C) {
	_asm {
		mov eax, B[0]
		push eax
		mov eax, A[0]
		push eax
		call pMultiplicacion
		fstp C[0]
		pop eax
		pop eax

		mov ebx, 15

		mul1:
			mov eax, B[4 * ebx]
			push eax
			mov eax, A[4 * ebx]
			push eax
			call pMultiplicacion
			fstp C[4 * ebx]
			pop eax
			pop eax

			dec ebx
			jnz mul1
	}
}

void div(float * A, float * B, float * C) {
	_asm {
		mov eax, B[0]
		push eax
		mov eax, A[0]
		push eax
		call pDivision
		fstp C[0]
		pop eax
		pop eax

		mov ebx, 15

		div1:
			mov eax, B[4 * ebx]
			push eax
			mov eax, A[4 * ebx]
			push eax
			call pDivision
			fstp C[4 * ebx]
			pop eax
			pop eax

			dec ebx
			jnz div1
	}
}