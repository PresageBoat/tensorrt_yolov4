#include "yololayer.h"
#include "yoloparam.h"

using namespace YoloParam;

namespace nvinfer1
{
	YoloLayerPlugin::YoloLayerPlugin()
	{
		mClassCount = CLASS_NUM;
		mYoloKernel.clear();
		mYoloKernel.push_back(yolo1);
		mYoloKernel.push_back(yolo2);
		mYoloKernel.push_back(yolo3);

		mKernelCount = mYoloKernel.size();
	}

	YoloLayerPlugin::~YoloLayerPlugin()
	{
	}

	// create the plugin at runtime from a byte stream
	YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
	{
		using namespace Tn;
		const char *d = reinterpret_cast<const char *>(data), *a = d;
		read(d, mClassCount);
		read(d, mThreadCount);
		read(d, mKernelCount);
		mYoloKernel.resize(mKernelCount);
		auto kernelSize = mKernelCount * sizeof(YoloKernel);
		memcpy(mYoloKernel.data(), d, kernelSize);
		d += kernelSize;

		assert(d == a + length);
	}

	void YoloLayerPlugin::serialize(void* buffer) const
	{
		using namespace Tn;
		char* d = static_cast<char*>(buffer), *a = d;
		write(d, mClassCount);
		write(d, mThreadCount);
		write(d, mKernelCount);
		auto kernelSize = mKernelCount * sizeof(YoloKernel);
		memcpy(d, mYoloKernel.data(), kernelSize);
		d += kernelSize;

		assert(d == a + getSerializationSize());
	}

	size_t YoloLayerPlugin::getSerializationSize() const
	{
		return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(YoloParam::YoloKernel) * mYoloKernel.size();
	}

	int YoloLayerPlugin::initialize()
	{
		return 0;
	}

	Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
	{
		//output the result to channel
		int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

		return Dims3(totalsize + 1, 1, 1);
	}

	// Set plugin namespace
	void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace)
	{
		mPluginNamespace = pluginNamespace;
	}

	const char* YoloLayerPlugin::getPluginNamespace() const
	{
		return mPluginNamespace;
	}

	// Return the DataType of the plugin output at the requested index
	DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
	{
		return DataType::kFLOAT;
	}

	// Return true if output tensor is broadcast across a batch.
	bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
	{
		return false;
	}

	// Return true if plugin can use input that is broadcast across batch without replication.
	bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
	{
		return false;
	}

	void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
	{
	}

	// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
	void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
	{
	}

	// Detach the plugin object from its execution context.
	void YoloLayerPlugin::detachFromContext() {}

	const char* YoloLayerPlugin::getPluginType() const
	{
		return "YoloLayer_TRT";
	}

	const char* YoloLayerPlugin::getPluginVersion() const
	{
		return "1";
	}

	void YoloLayerPlugin::destroy()
	{
		delete this;
	}

	// Clone the plugin
	IPluginV2IOExt* YoloLayerPlugin::clone() const
	{
		YoloLayerPlugin *p = new YoloLayerPlugin();
		p->setPluginNamespace(mPluginNamespace);
		return p;
	}



	PluginFieldCollection YoloPluginCreator::mFC{};
	std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

	YoloPluginCreator::YoloPluginCreator()
	{
		mPluginAttributes.clear();

		mFC.nbFields = mPluginAttributes.size();
		mFC.fields = mPluginAttributes.data();
	}

	const char* YoloPluginCreator::getPluginName() const
	{
		return "YoloLayer_TRT";
	}

	const char* YoloPluginCreator::getPluginVersion() const
	{
		return "1";
	}

	const PluginFieldCollection* YoloPluginCreator::getFieldNames()
	{
		return &mFC;
	}

	IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
	{
		YoloLayerPlugin* obj = new YoloLayerPlugin();
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}

	IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
	{
		// This object will be deleted when the network is destroyed, which will
		// call MishPlugin::destroy()
		YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}

}