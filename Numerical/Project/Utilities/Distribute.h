#pragma once

typedef struct{
    int start;
    int size;
} rnk_info;

rnk_info* distributeMatrix(int rank_size, int matrix_size);
