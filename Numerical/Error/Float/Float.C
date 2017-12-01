#include <iostream>

using namespace std;

float conver_to_float(int number){
    float * a = (float*) &number;
    return *a;
}

int main(int argc, char *argv[]){
    int pos_max_norm = 0b01111111011111111111111111111111; //2139095039;
    int neg_max_norm = 0b11111111011111111111111111111111; //4286578687;

    int pos_min_norm = 0b00000000100000000000000000000000; //8388608;
    int neg_min_norm = 0b10000000100000000000000000000000; //2155872256;

    int pos_zero = 0b00000000000000000000000000000000; //0;
    int neg_zero = 0b10000000000000000000000000000000; //2147483648;

    int pos_max_denormal = 0b00000000011111111111111111111111; //1;
    int neg_max_denormal = 0b10000000011111111111111111111111; //1;

    int pos_min_denormal = 0b00000000000000000000000000000001; //2147483649;
    int neg_min_denormal = 0b10000000000000000000000000000001; //2147483649;

    int pos_inf = 0b01111111100000000000000000000000; //2139095040;
    int neg_inf = 0b11111111100000000000000000000000; //4286578688;

    int pos_nan = 0b01111111100000000000000000000001; //2139095041;
    int neg_nan = 0b11111111100000000000000000000001; //4286578689;

    cout << "Positive max normal = " << conver_to_float(pos_max_norm) << endl;
    cout << "Negative max normal = " << conver_to_float(neg_max_norm) << endl;

    cout << "Positive min normal = " << conver_to_float(pos_min_norm) << endl;
    cout << "Negative min normal = " << conver_to_float(neg_min_norm) << endl;

    cout << "Positive zero = " << conver_to_float(pos_zero) << endl;
    cout << "Negative zero = " << conver_to_float(pos_zero) << endl;

    cout << "Positive max denormal = " << conver_to_float(pos_max_denormal) << endl;
    cout << "Negative max denormal = " << conver_to_float(neg_max_denormal) << endl;

    cout << "Positive min denormal = " << conver_to_float(pos_min_denormal) << endl;
    cout << "Negative min denormal = " << conver_to_float(neg_min_denormal) << endl;

    cout << "Positive infinite = " << conver_to_float(pos_inf) << endl;
    cout << "Negative infinite = " << conver_to_float(neg_inf) << endl;

    cout << "Positive nan = " << conver_to_float(pos_nan) << endl;
    cout << "Negative nan = " << conver_to_float(neg_nan) << endl;

    return 0;
}
