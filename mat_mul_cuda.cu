#include <iostream>
#include <vector>
#include <chrono>
#include <random>
using namespace std;

constexpr size_t N_INPUTS = 128*1024;
constexpr size_t N_NODES = 1024*1024;

template<typename T>
T random(T range_from, T range_to) {
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_real_distribution<T>    distr(range_from, range_to);
	return distr(generator);
}

__global__ void
dotprod(const float *const mat1, const float *const mat2, float *const dotprods) {

    // Use the thread ID as the node ID.
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    // mat2_node points to the beginning of the elements for this node (each thread handles a node).
    const float *const mat2_node = mat2 + id*N_INPUTS;

    // Compute dot product for this node.
    float dp = 0;
    for (size_t i = 0; i < N_INPUTS; i++) {
        dp += mat2_node[i]*mat1[i];
    }
    dotprods[id] = dp;
}

void initialize_vec(vector<float> &src)
{
    float a = 1.0;
    float b = 11.0;
    for(int i=0; i<src.size(); i++)
    {
        
        src[i] = random<float>(a, b);
    }
}

int main()
{
    vector<float> mat1(N_INPUTS);
    initialize_vec(mat1);

    vector<float> mat2 (N_INPUTS);
    initialize_vec(mat2);

    float *mat1_ptr;
    cudaMalloc(&mat1_ptr, N_INPUTS*sizeof(float));
    cudaMemcpy(mat1_ptr, mat1.data(), N_INPUTS*sizeof(float), cudaMemcpyHostToDevice);

    float *mat2_ptr;
    cudaMalloc(&mat2_ptr, N_INPUTS*sizeof(float));
    cudaMemcpy(mat2_ptr, mat2.data(), N_INPUTS*sizeof(float), cudaMemcpyHostToDevice);

    float *dotprod_ptr;
    cudaMalloc(&dotprod_ptr, N_NODES*sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    dotprod<<<N_NODES/1024, 1024>>>(mat1_ptr, mat2_ptr, dotprod_ptr);
    if(cudaDeviceSynchronize())
        cout<< "Synced!" <<endl;

    vector<float>res(N_NODES, 0.0);
    cudaMemcpy(res.data(), dotprod_ptr, N_NODES*sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);

    cout<< "Time Taken for CUDA Mat-Mul: " << duration.count() << " microseconds" <<endl;
}