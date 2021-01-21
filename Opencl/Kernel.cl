__constant char msg[] = "hello\n";

__kernel void hello(__global char * out) {
    size_t tid = get_global_id(0);
    out[tid] = msg[tid];
}
