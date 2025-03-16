- naive gemm
```C++
struct Matrix {
	int width;
	int height;
	float* elements;
	Matrix(Matrix o) : width(o.width), height(o.height){}
};

#define BLOCK_1D_SIZE 16

__global__ void MatMulKernel(const Matrix A, const Matrix B,  Matrix C)
{
	float cvalue = 0;
	int colId = blockIdx.x * blockDim.x + threadIdx.x; // cautious!
	int rowId = blockIdx.y * blockDim.y + threadIdx.y; // cautious!
	for (int e = 0; e < A.width; e++) {
		cvalue += A.elements[rowId * A.width + e] * B.elements[e * B.width + colId];	
	}
	c[rowId * C.width + colId] = cvalue;
}

void MatMul(const Matrix A, const Matrix B, const Matrix C) 
{
	assert(C.width  % BLOCK_1D_SIZE == 0);
	assert(C.height % BLOCK_1D_SIZE == 0);
	assert(C.width  == B.width);
	assert(C.height == A.height);
	// malloc mem for A, B, C
	Matrix d_A(A), d_B(B), d_C(C);
	cudaMalloc(&d_A.elements, sizeof(float)*A.width*A.height);
	cudaMalloc(&d_B.elements, sizeof(float)*B.width*B.height);
	cudaMalloc(&d_C.elements, sizeof(float)*C.width*C.height);

	// memcpy 
	cudaMemcpy(d_A.elements, A.elements, sizeof(float)*A.width*A.height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.elements, B.elements, sizeof(float)*B.width*B.height, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_1D_SIZE,BLOCK_1D_SIZE);
	dim3 dimGrid(C.width / BLOCK_1D_SIZE, C.height / BLOCK_1D_SIZE);
	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_b, d_C);

	// copyBack
	cudaMemcpy(C.elements, d_C.elements, sizeof(float)*B.width*B.height, cudaMemcpyDeviceToHost);
	// free
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}
```


- simple shared memory
```C++
#define BLOCK_1D_SIZE 16

struct Matrix {
	int width;
	int height;
	int stride;
	float* elements;
	Matrix(Matrix o) : width(o.width), height(o.height){}
	__device__ Matrix subMatrix(int blockRow, int blockCol) {
		Matrix sub;
		sub.width = BLOCK_1D_SIZE;
		sub.height = BLOCK_1D_SIZE;
		sub.stride = m.stride;
		sub.elements = m.elements + blockRow * BLOCK_1D_SIZE * m.stride + blockCol + BLOCK_1D_SIZE;
		return sub;
	}

	__device__ float getElement(int row, int col) {
		return elements[row * stride + col];
	}

	__device__ void setElement(float val, int row, int col) {
		element[row * stride + col] = val;
	}
};


__global__ void MatMulKernel(const Matrix A, const Matrix B,  Matrix C)
{
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;
	int threadRow = threadIdx.y;
	int threadCol = threadIdx.x;
	
	
	int numRounds = A.width / BLOCK_1D_SIZE;
	float cvalue = 0;
	for (int round = 0; round < numRounds; round++) {
		Matrix subA = A.subMatrix(blockRow, round);
		Matrix subB = B.subMatrix(round, blockCol);

		__shared__ sSubA[BLOCK_1D_SIZE][BLOCK_1D_SIZE];
		__shared__ sSubB[BLOCK_1D_SIZE][BLOCK_1D_SIZE];

		sSubA[threadRow][threadCol] = subA.getElement(threadRow,threadCol);
		sSubB[threadRow][threadCol] = subB.getElement(threadRow,threadCol);
		__syncthreads();

		for (int e = 0; e < BLOCK_1D_SIZE; e++) {
			cvalue += sSubA[threadRow][e] * sSubB[e][threadCol];
		}
		__syncthreads();
	}
	Matrix subC = C.subMatrix(blockRow, blockCol);
	subC.setElement(cvalue, threadRow, threadCol);
}


```