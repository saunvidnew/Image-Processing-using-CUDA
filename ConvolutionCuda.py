#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>
#made cahnges
#define MASK_WIDTH 5
#define TILE_WIDTH 16
#define CLAMP(x) (min(max((x), 0.0), 1.0))

__global__ void tiledConvolution2D(float* d_inputImage, float* d_outputImage,
                                   const float* __restrict__ d_mask,
                                   int channels, int width, int height) {
    __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int outputRow = by * TILE_WIDTH + ty;
    int outputCol = bx * TILE_WIDTH + tx;

    int inputRow = outputRow - MASK_WIDTH / 2;
    int inputCol = outputCol - MASK_WIDTH / 2;

    for (int ch = 0; ch < channels; ch++) {
        if (inputRow >= 0 && inputRow < height && inputCol >= 0 && inputCol < width) {
            tile[ty][tx][ch] = d_inputImage[(inputRow * width + inputCol) * channels + ch];
        } else {
            tile[ty][tx][ch] = 0.0f;
        }
    }
    __syncthreads();

    if (ty < TILE_WIDTH && tx < TILE_WIDTH && outputRow < height && outputCol < width) {
        for (int ch = 0; ch < channels; ch++) {
            float result = 0.0f;
            for (int i = 0; i < MASK_WIDTH; i++) {
                for (int j = 0; j < MASK_WIDTH; j++) {
                    result += d_mask[i * MASK_WIDTH + j] * tile[ty + i][tx + j][ch];
                }
            }
            d_outputImage[(outputRow * width + outputCol) * channels + ch] = CLAMP(result);
        }
    }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows, maskCols;
    int width, height, channels;
    char* inputImageFile;
    char* inputMaskFile;

    float* h_inputImage;
    float* h_outputImage;
    float* h_mask;
    float* d_inputImage;
    float* d_outputImage;
    float* d_mask;

    wbImage_t inputImg;
    wbImage_t outputImg;

    args = wbArg_read(argc, argv);
    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImg = wbImport(inputImageFile);
    h_mask = (float*)wbImport(inputMaskFile, &maskRows, &maskCols);

    assert(maskRows == MASK_WIDTH);
    assert(maskCols == MASK_WIDTH);

    width = wbImage_getWidth(inputImg);
    height = wbImage_getHeight(inputImg);
    channels = wbImage_getChannels(inputImg);

    outputImg = wbImage_new(width, height, channels);
    h_inputImage = wbImage_getData(inputImg);
    h_outputImage = wbImage_getData(outputImg);

    wbTime_start(GPU, "GPU Memory Allocation + Copy + Compute");

    // Allocate GPU memory
    cudaMalloc((void**)&d_inputImage, width * height * channels * sizeof(float));
    cudaMalloc((void**)&d_outputImage, width * height * channels * sizeof(float));
    cudaMalloc((void**)&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_inputImage, h_inputImage, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    // Define kernel configuration
    dim3 dimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, 1);
    dim3 dimGrid((width - 1) / TILE_WIDTH + 1, (height - 1) / TILE_WIDTH + 1, 1);

    // Launch kernel
    tiledConvolution2D<<<dimGrid, dimBlock>>>(d_inputImage, d_outputImage, d_mask, channels, width, height);

    // Copy result back to host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * channels * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(GPU, "GPU Memory Allocation + Copy + Compute");

    wbSolution(args, outputImg);

    // Free memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_mask);

    free(h_mask);
    wbImage_delete(inputImg);
    wbImage_delete(outputImg);

    return 0;
}
