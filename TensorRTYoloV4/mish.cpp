#include "mish.h"
#include <cmath>
#include <stdio.h>
#include <cassert>
#include <iostream>

namespace nvinfer1
{

	MishPlugin::MishPlugin()
	{
	}

	MishPlugin::~MishPlugin()
	{
	}

	// create the plugin at runtime from a byte stream
	MishPlugin::MishPlugin(const void* data, size_t length)
	{
		assert(length == sizeof(input_size_));
		input_size_ = *reinterpret_cast<const int*>(data);
	}

	void MishPlugin::serialize(void* buffer) const
	{
		*reinterpret_cast<int*>(buffer) = input_size_;
	}

	size_t MishPlugin::getSerializationSize() const
	{
		return sizeof(input_size_);
	}

	int MishPlugin::initialize()
	{
		return 0;
	}

	Dims MishPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
	{
		assert(nbInputDims == 1);
		assert(index == 0);
		input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
		// Output dimensions
		return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
	}

	// Set plugin namespace
	void MishPlugin::setPluginNamespace(const char* pluginNamespace)
	{
		mPluginNamespace = pluginNamespace;
	}

	const char* MishPlugin::getPluginNamespace() const
	{
		return mPluginNamespace;
	}

	// Return the DataType of the plugin output at the requested index
	DataType MishPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
	{
		return DataType::kFLOAT;
	}

	// Return true if output tensor is broadcast across a batch.
	bool MishPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
	{
		return false;
	}

	// Return true if plugin can use input that is broadcast across batch without replication.
	bool MishPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
	{
		return false;
	}

	void MishPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
	{
	}

	// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
	void MishPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
	{
	}

	// Detach the plugin object from its execution context.
	void MishPlugin::detachFromContext() {}

	const char* MishPlugin::getPluginType() const
	{
		return "Mish_TRT";
	}

	const char* MishPlugin::getPluginVersion() const
	{
		return "1";
	}

	void MishPlugin::destroy()
	{
		delete this;
	}

	// Clone the plugin
	IPluginV2IOExt* MishPlugin::clone() const
	{
		MishPlugin *p = new MishPlugin();
		p->input_size_ = input_size_;
		p->setPluginNamespace(mPluginNamespace);
		return p;
	}



	//MishPluginCreator class 
	PluginFieldCollection MishPluginCreator::mFC{};
	std::vector<PluginField> MishPluginCreator::mPluginAttributes;

	MishPluginCreator::MishPluginCreator()
	{
		mPluginAttributes.clear();

		mFC.nbFields = mPluginAttributes.size();
		mFC.fields = mPluginAttributes.data();
	}

	const char* MishPluginCreator::getPluginName() const
	{
		return "Mish_TRT";
	}

	const char* MishPluginCreator::getPluginVersion() const
	{
		return "1";
	}

	const PluginFieldCollection* MishPluginCreator::getFieldNames()
	{
		return &mFC;
	}

	IPluginV2IOExt* MishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
	{
		MishPlugin* obj = new MishPlugin();
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}

	IPluginV2IOExt* MishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
	{
		// This object will be deleted when the network is destroyed, which will
		// call MishPlugin::destroy()
		MishPlugin* obj = new MishPlugin(serialData, serialLength);
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}

}