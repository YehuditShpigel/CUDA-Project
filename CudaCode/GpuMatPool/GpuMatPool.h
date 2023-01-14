#pragma once
#include "opencv2/opencv.hpp"
#include <opencv2/core/cuda.hpp>

typedef std::shared_ptr<cv::cuda::GpuMat> GMatPtr;

class GpuMatPool
{
public:
	std::queue<GMatPtr> available;
	int _w, _h;
public:
	class GpuMatContainer {
	public:
		GMatPtr mat;
		GpuMatPool* _pool;
		GpuMatContainer(GpuMatPool* pool);

		~GpuMatContainer();
		operator cv::cuda::GpuMat& () { return *mat; }
		cv::cuda::GpuMat& m() { return *mat; }

	};
	GpuMatPool(int w, int h) { this->_w = w; this->_h = h; }

	std::shared_ptr<GpuMatContainer > get();

};
typedef std::shared_ptr<GpuMatPool::GpuMatContainer > GMatCPtr;