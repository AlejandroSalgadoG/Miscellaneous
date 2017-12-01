#include <stdio.h>
#include "../Debug/AssemblyFunctions.h"

char * formato = "%s";
char * msg = "Division by 0 detected, the position will be indicated with \"inf\"\n";

float pSuma(float a, float b) {
	__asm {
		fld a
		fadd b
	}
}

float pResta(float a, float b) {
	__asm {
		fld a
		fsub b
	}
}

float pMultiplicacion(float a, float b) {
	__asm {
		fld a
		fmul b
	}
}

float pDivision(float a, float b) {

	float c = 0.0f;

	__asm {
		fld b
		fld c
		fcompp
		fstsw ax
		sahf

		jz zero
		jnz divide

		zero:
			push formato
			push msg
			call printf
			pop eax
			pop eax

		divide:
			fld a
			fdiv b
	}
}