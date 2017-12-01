#include "Distribute.h"

rnk_info* distributeMatrix(int rank_size, int matrix_size){

    int mod = matrix_size % rank_size;
    int size = matrix_size / rank_size;

    rnk_info * info = new rnk_info[rank_size];

    info[0].start = 0;
    if(mod > 0){
        info[0].size = size + 1; mod--;
    }
    else info[0].size = size;

    for(int i=1;i<rank_size;i++){
        info[i].start = info[i-1].start + info[i-1].size;

        if(mod > 0){
            info[i].size = size + 1; mod--;
        }
        else if(mod == 0) info[i].size = size;
    }

    return info;
}
