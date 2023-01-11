#include "pch.h"
#include "GpuMatPool.h"

using namespace std;
using namespace cv::cuda;

GpuMatPool::GpuMatContainer::GpuMatContainer(GpuMatPool* pool)
{
	_pool = pool;
	if (_pool->available.empty())
		_pool->available.push(GMatPtr(new GpuMat(_pool->_w, _pool->_h, CV_32F)));


	mat = _pool->available.front();
	_pool->available.pop();
}
GpuMatPool::GpuMatContainer::~GpuMatContainer() {
	_pool->available.push(mat);
}
std::shared_ptr<GpuMatPool::GpuMatContainer> GpuMatPool::get()
{
	return std::make_shared<GpuMatContainer>(this);
}
