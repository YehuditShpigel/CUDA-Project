#pragma once
#include "../GpuMatPool/GpuMatPool.h"
#include "../CudaConvolution/kernel.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudafeatures2d.hpp>


GpuMatPool ThePool(1080, 1920);

std::vector<cv::cuda::GpuMat> matsPull;


GMatCPtr cudaAdd(cv::cuda::GpuMat src1, cv::cuda::GpuMat src2);
GMatCPtr cudaSubtract(cv::cuda::GpuMat src1, cv::cuda::GpuMat src2);
GMatCPtr cudaMultiply(cv::cuda::GpuMat src1, cv::cuda::GpuMat src2);
GMatCPtr cudaMultiply(cv::cuda::GpuMat src1, float src2);
GMatCPtr cudaDivide(cv::cuda::GpuMat src1, cv::cuda::GpuMat src2);
GMatCPtr cudaTranspose(cv::cuda::GpuMat src);
GMatCPtr cudaAbs(cv::cuda::GpuMat src);
GMatCPtr conv(const cv::cuda::GpuMat src, const cv::cuda::GpuMat kernel);
GMatCPtr specificConv(const cv::cuda::GpuMat src, const float* kernel_ptr, int kernelH, int kernelW);
