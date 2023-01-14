#include "OpencvCudaUtils.h"

using namespace cv;
using namespace cuda;

GMatCPtr cudaAdd(GpuMat src1, GpuMat src2) {
	auto matContainer = ThePool.get();
	cuda::add(src1, src2, matContainer->m());
	return matContainer;
}
GMatCPtr cudaSubtract(GpuMat src1, GpuMat src2) {
	/*GpuMat dst;
	cuda::subtract(src1, src2, dst);
	return dst;*/
	//int matIndex = findMat();
	//cuda::subtract(src1, src2, matsPull[matIndex]);
	//return matsPull[matIndex];
	auto matContainer = ThePool.get();
	cuda::subtract(src1, src2, matContainer->m());
	return matContainer;
}
GMatCPtr cudaMultiply(GpuMat src1, GpuMat src2) {
	//GpuMat dst;
	//cuda::multiply(src1, src2, dst);
	//return dst;
	//int matIndex = findMat();
	//cuda::multiply(src1, src2, matsPull[matIndex]);
	//return matsPull[matIndex];
	auto matContainer = ThePool.get();
	cuda::multiply(src1, src2, matContainer->m());
	return matContainer;
}
GMatCPtr cudaMultiply(GpuMat src1, float src2) {
	//GpuMat dst;
	//cuda::multiply(src1, src2, dst);
	//return dst;
	//int matIndex = findMat();
	//cuda::multiply(src1, src2, matsPull[matIndex]);
	//return matsPull[matIndex];
	auto matContainer = ThePool.get();
	cuda::multiply(src1, src2, matContainer->m());
	return matContainer;
}
GMatCPtr cudaDivide(GpuMat src1, GpuMat src2) {
	/*GpuMat dst;
	cuda::divide(src1, src2, dst);
	return dst;*/
	//int matIndex = findMat();
	//cuda::divide(src1, src2, matsPull[matIndex]);
	//return matsPull[matIndex];
	auto matContainer = ThePool.get();
	cuda::divide(src1, src2, matContainer->m());
	return matContainer;
}
GMatCPtr cudaTranspose(GpuMat src) {
	/*GpuMat dst;
	cuda::transpose(src, dst);
	return dst;*/
	//	int matIndex = findMat();
	//	cuda::transpose(src, matsPull[matIndex]);
	//	return matsPull[matIndex];
	auto matContainer = ThePool.get();
	cuda::transpose(src, matContainer->m());
	return matContainer;
}
GMatCPtr cudaAbs(GpuMat src) {
	auto matContainer = ThePool.get();
	cuda::abs(src, matContainer->m());
	return matContainer;
}
GMatCPtr conv(const GpuMat src, const GpuMat kernel) {

	auto matContainer = ThePool.get();
	auto dst = matContainer->m();
	//GpuMat dst(src.rows, src.cols, src.type());
		//GpuMat dst = matsPull[findMat()];
		/*Ptr<cuda::Filter> filter = cuda::createLinearFilter(CV_32FC1, CV_32FC1, kernel, Point(-1, -1), BORDER_CONSTANT);
		filter->apply(src, dst);*/

		/*GpuMat biggerSrc;
		cuda::copyMakeBorder(src, biggerSrc, kernel.rows / 2, kernel.rows / 2, kernel.cols / 2, kernel.cols / 2, BORDER_CONSTANT, 0);
		Ptr<cuda::Convolution> convolver = cuda::createConvolution(Size(0, 0));
		convolver->convolve(biggerSrc, kernel, dst);*/

		/*Mat srcMat, kernelMat, cudaMat;
		src.download(srcMat);
		kernel.download(kernelMat);
		Mat dstMat(srcMat.rows, srcMat.cols, srcMat.type());
		Mat dstMatCuda(srcMat.rows, srcMat.cols, srcMat.type());
		dstMat = conv(srcMat, kernelMat);
		dst.upload(dstMat);*/

	convolvWithCuda((float*)(matContainer->m()).data, (float*)src.data, (float*)kernel.data, src.cols, src.rows, kernel.cols, kernel.rows);// , (float*)dstMatCuda.data, (float*)srcMat.data, (float*)kernelMat.data);
	return matContainer;

	//dst.download(cudaMat);
	//Mat diff = cudaMat - dstMat;
	//Mat diff = dstMatCuda - dstMat;

	//return dst;
}
GMatCPtr specificConv(const GpuMat src, const float* kernel_ptr, int kernelH, int kernelW) {
	//GpuMat dst(src.rows, src.cols, src.type());
	//GpuMat dst = matsPull[findMat()];

	//Mat srcMat, kernelMat, cudaMat;
	//src.download(srcMat);
	/*kernel.download(kernelMat);
	{
		auto filterHeight = kernelMat.rows;
		auto filterWidth = kernelMat.cols;
		float* kernelTest = (float*)malloc(sizeof(float) * filterHeight * filterWidth);
		auto cudaStatus = cudaMemcpy(kernelTest, kernel.ptr(), filterHeight * filterWidth * sizeof(float), cudaMemcpyDeviceToHost);
		Mat kernelTestMat(filterHeight, filterWidth, CV_32F, kernelTest);
		__debugbreak();

	}*/
	//Mat dstMat(srcMat.rows, srcMat.cols, srcMat.type());
	//Mat dstMatCuda(srcMat.rows, srcMat.cols, srcMat.type());
	//dstMat = conv(srcMat, kernelMat);
	auto matContainer = ThePool.get();
	auto dst = matContainer->m();
	convolvWithCuda((matContainer->m()).ptr<float>(), src.ptr<float>(), kernel_ptr, src.cols, src.rows, kernelW, kernelH);// , dstMatCuda.ptr<float>(), srcMat.ptr<float>(), kernelMat.ptr<float>());
	return matContainer;
	//dst.download(cudaMat);
	//dst.upload(dstMat);
	//Mat diff = dstMatCuda - dstMat;

	//return dst;
}