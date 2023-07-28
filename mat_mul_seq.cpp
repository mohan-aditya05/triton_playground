#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>

using namespace std;

vector<vector<float>> dot_prod(vector<vector<float>>& mat1, vector<vector<float>>& mat2)
{
    assert(mat1[0].size() == mat2.size());
    vector<vector<float>> result(mat1.size(), vector<float>(mat2[0].size(), 0.0));
    for(int i=0; i<mat1.size(); i++)
    {
        for(int j=0; j<mat2[0].size(); j++)
        {
            for(int k=0; k<mat2.size(); k++)
            {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

template<typename T>
T random(T range_from, T range_to) {
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_real_distribution<T>    distr(range_from, range_to);
	return distr(generator);
}

void initialize_vec(vector<vector<float>> &src)
{
    float a = 1.0;
    float b = 11.0;
    for(int i=0; i<src.size(); i++)
    {
        for(int j=0; j<src[0].size(); j++)
        {
            src[i][j] = random<float>(a, b);
        }
    }
}

int main()
{
    vector<vector<float>> mat1(1024, vector<float>(128, 0.0));
    vector<vector<float>> mat2(128, vector<float>(1024, 0.0));

    initialize_vec(mat1);
    initialize_vec(mat2);

    auto start = chrono::high_resolution_clock::now();
    auto res = dot_prod(mat1, mat2);
    auto end = chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);

    cout<< "Time Taken for Sequential Mat-Mul: " << duration.count() << " microseconds" <<endl;

}