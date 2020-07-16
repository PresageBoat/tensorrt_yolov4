#include <iostream>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include "../TensorRTYoloV4/logging.h"
#include "../TensorRTYoloV4/yololayer.h"
#include "../TensorRTYoloV4/mish.h"
#include "./calibrator.h"

#define DEVICE 0 //GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5
#define BATCH_SIZE 128
#define CALIB_IMAGEDIR "../model/calibratorimages"
#define CALIBTABLE_PATH "../model/yolov4-coco/calibration.table"

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 416;
static const int INPUT_W = 416;
static const int INPUT_C = 3;
static const int DETECTION_SIZE = sizeof(YoloParam::Detection) / sizeof(float); //7
// we assume the yolo layer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static const int OUTPUT_SIZE = YoloParam::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

std::map<std::string, Weights> loadWeights(const std::string file)
{
	std::cout << "Loading weights: " << file << std::endl;
	std::map<std::string, Weights> weightMap;

	// Open weights file
	std::ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--)
	{
		Weights wt{DataType::kFLOAT, nullptr, 0};
		uint32_t size;

		// Read name and type of blob
		std::string name;
		input >> name >> std::dec >> size;
		wt.type = DataType::kFLOAT;

		// Load blob
		uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> std::hex >> val[x];
		}
		wt.values = val;

		wt.count = size;
		weightMap[name] = wt;
	}

	return weightMap;
}

IScaleLayer *AddBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
{
	float *gamma = (float *)weightMap[lname + ".weight"].values;
	float *beta = (float *)weightMap[lname + ".bias"].values;
	float *mean = (float *)weightMap[lname + ".running_mean"].values;
	float *var = (float *)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;
	//std::cout << "len " << len << std::endl;

	float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++)
	{
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{DataType::kFLOAT, scval, len};

	float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++)
	{
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{DataType::kFLOAT, shval, len};

	float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++)
	{
		pval[i] = 1.0;
	}
	Weights power{DataType::kFLOAT, pval, len};

	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
	return scale_1;
}

ILayer *ConvBnMish(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize, int s, int p, int linx)
{
	//std::cout << linx << std::endl;
	Weights emptywts{DataType::kFLOAT, nullptr, 0};
	IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{s, s});
	conv1->setPaddingNd(DimsHW{p, p});

	IScaleLayer *bn1 = AddBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

	auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
	const PluginFieldCollection *pluginData = creator->getFieldNames();
	IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(linx)).c_str(), pluginData);
	ITensor *inputTensors[] = {bn1->getOutput(0)};
	auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
	return mish;
}

ILayer *ConvBnLeaky(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize, int s, int p, int linx)
{
	//std::cout << linx << std::endl;
	Weights emptywts{DataType::kFLOAT, nullptr, 0};
	IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{s, s});
	conv1->setPaddingNd(DimsHW{p, p});

	IScaleLayer *bn1 = AddBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

	auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
	lr->setAlpha(0.1);

	return lr;
}

//Creat the engine using only the API and not any parser.
ICudaEngine *createEngine(IBuilder *builder, IBuilderConfig *config,
						  DataType dtype, const std::string wtspath)
{

	INetworkDefinition *network = builder->createNetworkV2(0U);

	// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
	ITensor *data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, Dims3{3, INPUT_H, INPUT_W});
	assert(data);

	std::map<std::string, Weights> weightMap = loadWeights(wtspath);
	Weights emptywts{DataType::kFLOAT, nullptr, 0};

	// define each layer.
	auto l0 = ConvBnMish(network, weightMap, *data, 32, 3, 1, 1, 0);
	auto l1 = ConvBnMish(network, weightMap, *l0->getOutput(0), 64, 3, 2, 1, 1);
	auto l2 = ConvBnMish(network, weightMap, *l1->getOutput(0), 64, 1, 1, 0, 2);
	auto l3 = l1;
	auto l4 = ConvBnMish(network, weightMap, *l3->getOutput(0), 64, 1, 1, 0, 4);
	auto l5 = ConvBnMish(network, weightMap, *l4->getOutput(0), 32, 1, 1, 0, 5);
	auto l6 = ConvBnMish(network, weightMap, *l5->getOutput(0), 64, 3, 1, 1, 6);
	auto ew7 = network->addElementWise(*l6->getOutput(0), *l4->getOutput(0), ElementWiseOperation::kSUM);
	auto l8 = ConvBnMish(network, weightMap, *ew7->getOutput(0), 64, 1, 1, 0, 8);

	ITensor *inputTensors9[] = {l8->getOutput(0), l2->getOutput(0)};
	auto cat9 = network->addConcatenation(inputTensors9, 2);

	auto l10 = ConvBnMish(network, weightMap, *cat9->getOutput(0), 64, 1, 1, 0, 10);
	auto l11 = ConvBnMish(network, weightMap, *l10->getOutput(0), 128, 3, 2, 1, 11);
	auto l12 = ConvBnMish(network, weightMap, *l11->getOutput(0), 64, 1, 1, 0, 12);
	auto l13 = l11;
	auto l14 = ConvBnMish(network, weightMap, *l13->getOutput(0), 64, 1, 1, 0, 14);
	auto l15 = ConvBnMish(network, weightMap, *l14->getOutput(0), 64, 1, 1, 0, 15);
	auto l16 = ConvBnMish(network, weightMap, *l15->getOutput(0), 64, 3, 1, 1, 16);
	auto ew17 = network->addElementWise(*l16->getOutput(0), *l14->getOutput(0), ElementWiseOperation::kSUM);
	auto l18 = ConvBnMish(network, weightMap, *ew17->getOutput(0), 64, 1, 1, 0, 18);
	auto l19 = ConvBnMish(network, weightMap, *l18->getOutput(0), 64, 3, 1, 1, 19);
	auto ew20 = network->addElementWise(*l19->getOutput(0), *ew17->getOutput(0), ElementWiseOperation::kSUM);
	auto l21 = ConvBnMish(network, weightMap, *ew20->getOutput(0), 64, 1, 1, 0, 21);

	ITensor *inputTensors22[] = {l21->getOutput(0), l12->getOutput(0)};
	auto cat22 = network->addConcatenation(inputTensors22, 2);

	auto l23 = ConvBnMish(network, weightMap, *cat22->getOutput(0), 128, 1, 1, 0, 23);
	auto l24 = ConvBnMish(network, weightMap, *l23->getOutput(0), 256, 3, 2, 1, 24);
	auto l25 = ConvBnMish(network, weightMap, *l24->getOutput(0), 128, 1, 1, 0, 25);
	auto l26 = l24;
	auto l27 = ConvBnMish(network, weightMap, *l26->getOutput(0), 128, 1, 1, 0, 27);
	auto l28 = ConvBnMish(network, weightMap, *l27->getOutput(0), 128, 1, 1, 0, 28);
	auto l29 = ConvBnMish(network, weightMap, *l28->getOutput(0), 128, 3, 1, 1, 29);
	auto ew30 = network->addElementWise(*l29->getOutput(0), *l27->getOutput(0), ElementWiseOperation::kSUM);
	auto l31 = ConvBnMish(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
	auto l32 = ConvBnMish(network, weightMap, *l31->getOutput(0), 128, 3, 1, 1, 32);
	auto ew33 = network->addElementWise(*l32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
	auto l34 = ConvBnMish(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
	auto l35 = ConvBnMish(network, weightMap, *l34->getOutput(0), 128, 3, 1, 1, 35);
	auto ew36 = network->addElementWise(*l35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
	auto l37 = ConvBnMish(network, weightMap, *ew36->getOutput(0), 128, 1, 1, 0, 37);
	auto l38 = ConvBnMish(network, weightMap, *l37->getOutput(0), 128, 3, 1, 1, 38);
	auto ew39 = network->addElementWise(*l38->getOutput(0), *ew36->getOutput(0), ElementWiseOperation::kSUM);
	auto l40 = ConvBnMish(network, weightMap, *ew39->getOutput(0), 128, 1, 1, 0, 40);
	auto l41 = ConvBnMish(network, weightMap, *l40->getOutput(0), 128, 3, 1, 1, 41);
	auto ew42 = network->addElementWise(*l41->getOutput(0), *ew39->getOutput(0), ElementWiseOperation::kSUM);
	auto l43 = ConvBnMish(network, weightMap, *ew42->getOutput(0), 128, 1, 1, 0, 43);
	auto l44 = ConvBnMish(network, weightMap, *l43->getOutput(0), 128, 3, 1, 1, 44);
	auto ew45 = network->addElementWise(*l44->getOutput(0), *ew42->getOutput(0), ElementWiseOperation::kSUM);
	auto l46 = ConvBnMish(network, weightMap, *ew45->getOutput(0), 128, 1, 1, 0, 46);
	auto l47 = ConvBnMish(network, weightMap, *l46->getOutput(0), 128, 3, 1, 1, 47);
	auto ew48 = network->addElementWise(*l47->getOutput(0), *ew45->getOutput(0), ElementWiseOperation::kSUM);
	auto l49 = ConvBnMish(network, weightMap, *ew48->getOutput(0), 128, 1, 1, 0, 49);
	auto l50 = ConvBnMish(network, weightMap, *l49->getOutput(0), 128, 3, 1, 1, 50);
	auto ew51 = network->addElementWise(*l50->getOutput(0), *ew48->getOutput(0), ElementWiseOperation::kSUM);
	auto l52 = ConvBnMish(network, weightMap, *ew51->getOutput(0), 128, 1, 1, 0, 52);

	ITensor *inputTensors53[] = {l52->getOutput(0), l25->getOutput(0)};
	auto cat53 = network->addConcatenation(inputTensors53, 2);

	auto l54 = ConvBnMish(network, weightMap, *cat53->getOutput(0), 256, 1, 1, 0, 54);
	auto l55 = ConvBnMish(network, weightMap, *l54->getOutput(0), 512, 3, 2, 1, 55);
	auto l56 = ConvBnMish(network, weightMap, *l55->getOutput(0), 256, 1, 1, 0, 56);
	auto l57 = l55;
	auto l58 = ConvBnMish(network, weightMap, *l57->getOutput(0), 256, 1, 1, 0, 58);
	auto l59 = ConvBnMish(network, weightMap, *l58->getOutput(0), 256, 1, 1, 0, 59);
	auto l60 = ConvBnMish(network, weightMap, *l59->getOutput(0), 256, 3, 1, 1, 60);
	auto ew61 = network->addElementWise(*l60->getOutput(0), *l58->getOutput(0), ElementWiseOperation::kSUM);
	auto l62 = ConvBnMish(network, weightMap, *ew61->getOutput(0), 256, 1, 1, 0, 62);
	auto l63 = ConvBnMish(network, weightMap, *l62->getOutput(0), 256, 3, 1, 1, 63);
	auto ew64 = network->addElementWise(*l63->getOutput(0), *ew61->getOutput(0), ElementWiseOperation::kSUM);
	auto l65 = ConvBnMish(network, weightMap, *ew64->getOutput(0), 256, 1, 1, 0, 65);
	auto l66 = ConvBnMish(network, weightMap, *l65->getOutput(0), 256, 3, 1, 1, 66);
	auto ew67 = network->addElementWise(*l66->getOutput(0), *ew64->getOutput(0), ElementWiseOperation::kSUM);
	auto l68 = ConvBnMish(network, weightMap, *ew67->getOutput(0), 256, 1, 1, 0, 68);
	auto l69 = ConvBnMish(network, weightMap, *l68->getOutput(0), 256, 3, 1, 1, 69);
	auto ew70 = network->addElementWise(*l69->getOutput(0), *ew67->getOutput(0), ElementWiseOperation::kSUM);
	auto l71 = ConvBnMish(network, weightMap, *ew70->getOutput(0), 256, 1, 1, 0, 71);
	auto l72 = ConvBnMish(network, weightMap, *l71->getOutput(0), 256, 3, 1, 1, 72);
	auto ew73 = network->addElementWise(*l72->getOutput(0), *ew70->getOutput(0), ElementWiseOperation::kSUM);
	auto l74 = ConvBnMish(network, weightMap, *ew73->getOutput(0), 256, 1, 1, 0, 74);
	auto l75 = ConvBnMish(network, weightMap, *l74->getOutput(0), 256, 3, 1, 1, 75);
	auto ew76 = network->addElementWise(*l75->getOutput(0), *ew73->getOutput(0), ElementWiseOperation::kSUM);
	auto l77 = ConvBnMish(network, weightMap, *ew76->getOutput(0), 256, 1, 1, 0, 77);
	auto l78 = ConvBnMish(network, weightMap, *l77->getOutput(0), 256, 3, 1, 1, 78);
	auto ew79 = network->addElementWise(*l78->getOutput(0), *ew76->getOutput(0), ElementWiseOperation::kSUM);
	auto l80 = ConvBnMish(network, weightMap, *ew79->getOutput(0), 256, 1, 1, 0, 80);
	auto l81 = ConvBnMish(network, weightMap, *l80->getOutput(0), 256, 3, 1, 1, 81);
	auto ew82 = network->addElementWise(*l81->getOutput(0), *ew79->getOutput(0), ElementWiseOperation::kSUM);
	auto l83 = ConvBnMish(network, weightMap, *ew82->getOutput(0), 256, 1, 1, 0, 83);

	ITensor *inputTensors84[] = {l83->getOutput(0), l56->getOutput(0)};
	auto cat84 = network->addConcatenation(inputTensors84, 2);

	auto l85 = ConvBnMish(network, weightMap, *cat84->getOutput(0), 512, 1, 1, 0, 85);
	auto l86 = ConvBnMish(network, weightMap, *l85->getOutput(0), 1024, 3, 2, 1, 86);
	auto l87 = ConvBnMish(network, weightMap, *l86->getOutput(0), 512, 1, 1, 0, 87);
	auto l88 = l86;
	auto l89 = ConvBnMish(network, weightMap, *l88->getOutput(0), 512, 1, 1, 0, 89);
	auto l90 = ConvBnMish(network, weightMap, *l89->getOutput(0), 512, 1, 1, 0, 90);
	auto l91 = ConvBnMish(network, weightMap, *l90->getOutput(0), 512, 3, 1, 1, 91);
	auto ew92 = network->addElementWise(*l91->getOutput(0), *l89->getOutput(0), ElementWiseOperation::kSUM);
	auto l93 = ConvBnMish(network, weightMap, *ew92->getOutput(0), 512, 1, 1, 0, 93);
	auto l94 = ConvBnMish(network, weightMap, *l93->getOutput(0), 512, 3, 1, 1, 94);
	auto ew95 = network->addElementWise(*l94->getOutput(0), *ew92->getOutput(0), ElementWiseOperation::kSUM);
	auto l96 = ConvBnMish(network, weightMap, *ew95->getOutput(0), 512, 1, 1, 0, 96);
	auto l97 = ConvBnMish(network, weightMap, *l96->getOutput(0), 512, 3, 1, 1, 97);
	auto ew98 = network->addElementWise(*l97->getOutput(0), *ew95->getOutput(0), ElementWiseOperation::kSUM);
	auto l99 = ConvBnMish(network, weightMap, *ew98->getOutput(0), 512, 1, 1, 0, 99);
	auto l100 = ConvBnMish(network, weightMap, *l99->getOutput(0), 512, 3, 1, 1, 100);
	auto ew101 = network->addElementWise(*l100->getOutput(0), *ew98->getOutput(0), ElementWiseOperation::kSUM);
	auto l102 = ConvBnMish(network, weightMap, *ew101->getOutput(0), 512, 1, 1, 0, 102);

	ITensor *inputTensors103[] = {l102->getOutput(0), l87->getOutput(0)};
	auto cat103 = network->addConcatenation(inputTensors103, 2);

	auto l104 = ConvBnMish(network, weightMap, *cat103->getOutput(0), 1024, 1, 1, 0, 104);

	// ---------
	auto l105 = ConvBnLeaky(network, weightMap, *l104->getOutput(0), 512, 1, 1, 0, 105);
	auto l106 = ConvBnLeaky(network, weightMap, *l105->getOutput(0), 1024, 3, 1, 1, 106);
	auto l107 = ConvBnLeaky(network, weightMap, *l106->getOutput(0), 512, 1, 1, 0, 107);

	auto pool108 = network->addPoolingNd(*l107->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
	pool108->setPaddingNd(DimsHW{2, 2});
	pool108->setStrideNd(DimsHW{1, 1});

	auto l109 = l107;

	auto pool110 = network->addPoolingNd(*l109->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
	pool110->setPaddingNd(DimsHW{4, 4});
	pool110->setStrideNd(DimsHW{1, 1});

	auto l111 = l107;

	auto pool112 = network->addPoolingNd(*l111->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
	pool112->setPaddingNd(DimsHW{6, 6});
	pool112->setStrideNd(DimsHW{1, 1});

	ITensor *inputTensors113[] = {pool112->getOutput(0), pool110->getOutput(0), pool108->getOutput(0), l107->getOutput(0)};
	auto cat113 = network->addConcatenation(inputTensors113, 4);

	auto l114 = ConvBnLeaky(network, weightMap, *cat113->getOutput(0), 512, 1, 1, 0, 114);
	auto l115 = ConvBnLeaky(network, weightMap, *l114->getOutput(0), 1024, 3, 1, 1, 115);
	auto l116 = ConvBnLeaky(network, weightMap, *l115->getOutput(0), 512, 1, 1, 0, 116);
	auto l117 = ConvBnLeaky(network, weightMap, *l116->getOutput(0), 256, 1, 1, 0, 117);

	float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 256 * 2 * 2));
	for (int i = 0; i < 256 * 2 * 2; i++)
	{
		deval[i] = 1.0;
	}
	Weights deconvwts118{DataType::kFLOAT, deval, 256 * 2 * 2};
	IDeconvolutionLayer *deconv118 = network->addDeconvolutionNd(*l117->getOutput(0), 256, DimsHW{2, 2}, deconvwts118, emptywts);
	assert(deconv118);
	deconv118->setStrideNd(DimsHW{2, 2});
	deconv118->setNbGroups(256);
	weightMap["deconv118"] = deconvwts118;

	auto l119 = l85;
	auto l120 = ConvBnLeaky(network, weightMap, *l119->getOutput(0), 256, 1, 1, 0, 120);

	ITensor *inputTensors121[] = {l120->getOutput(0), deconv118->getOutput(0)};
	auto cat121 = network->addConcatenation(inputTensors121, 2);

	auto l122 = ConvBnLeaky(network, weightMap, *cat121->getOutput(0), 256, 1, 1, 0, 122);
	auto l123 = ConvBnLeaky(network, weightMap, *l122->getOutput(0), 512, 3, 1, 1, 123);
	auto l124 = ConvBnLeaky(network, weightMap, *l123->getOutput(0), 256, 1, 1, 0, 124);
	auto l125 = ConvBnLeaky(network, weightMap, *l124->getOutput(0), 512, 3, 1, 1, 125);
	auto l126 = ConvBnLeaky(network, weightMap, *l125->getOutput(0), 256, 1, 1, 0, 126);
	auto l127 = ConvBnLeaky(network, weightMap, *l126->getOutput(0), 128, 1, 1, 0, 127);

	Weights deconvwts128{DataType::kFLOAT, deval, 128 * 2 * 2};
	IDeconvolutionLayer *deconv128 = network->addDeconvolutionNd(*l127->getOutput(0), 128, DimsHW{2, 2}, deconvwts128, emptywts);
	assert(deconv128);
	deconv128->setStrideNd(DimsHW{2, 2});
	deconv128->setNbGroups(128);

	auto l129 = l54;
	auto l130 = ConvBnLeaky(network, weightMap, *l129->getOutput(0), 128, 1, 1, 0, 130);

	ITensor *inputTensors131[] = {l130->getOutput(0), deconv128->getOutput(0)};
	auto cat131 = network->addConcatenation(inputTensors131, 2);

	auto l132 = ConvBnLeaky(network, weightMap, *cat131->getOutput(0), 128, 1, 1, 0, 132);
	auto l133 = ConvBnLeaky(network, weightMap, *l132->getOutput(0), 256, 3, 1, 1, 133);
	auto l134 = ConvBnLeaky(network, weightMap, *l133->getOutput(0), 128, 1, 1, 0, 134);
	auto l135 = ConvBnLeaky(network, weightMap, *l134->getOutput(0), 256, 3, 1, 1, 135);
	auto l136 = ConvBnLeaky(network, weightMap, *l135->getOutput(0), 128, 1, 1, 0, 136);
	auto l137 = ConvBnLeaky(network, weightMap, *l136->getOutput(0), 256, 3, 1, 1, 137);
	IConvolutionLayer *conv138 = network->addConvolutionNd(*l137->getOutput(0), 3 * (YoloParam::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.138.Conv2d.weight"], weightMap["module_list.138.Conv2d.bias"]);
	assert(conv138);
	// 139 is yolo layer

	auto l140 = l136;
	auto l141 = ConvBnLeaky(network, weightMap, *l140->getOutput(0), 256, 3, 2, 1, 141);

	ITensor *inputTensors142[] = {l141->getOutput(0), l126->getOutput(0)};
	auto cat142 = network->addConcatenation(inputTensors142, 2);

	auto l143 = ConvBnLeaky(network, weightMap, *cat142->getOutput(0), 256, 1, 1, 0, 143);
	auto l144 = ConvBnLeaky(network, weightMap, *l143->getOutput(0), 512, 3, 1, 1, 144);
	auto l145 = ConvBnLeaky(network, weightMap, *l144->getOutput(0), 256, 1, 1, 0, 145);
	auto l146 = ConvBnLeaky(network, weightMap, *l145->getOutput(0), 512, 3, 1, 1, 146);
	auto l147 = ConvBnLeaky(network, weightMap, *l146->getOutput(0), 256, 1, 1, 0, 147);
	auto l148 = ConvBnLeaky(network, weightMap, *l147->getOutput(0), 512, 3, 1, 1, 148);
	IConvolutionLayer *conv149 = network->addConvolutionNd(*l148->getOutput(0), 3 * (YoloParam::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.149.Conv2d.weight"], weightMap["module_list.149.Conv2d.bias"]);
	assert(conv149);
	// 150 is yolo layer

	auto l151 = l147;
	auto l152 = ConvBnLeaky(network, weightMap, *l151->getOutput(0), 512, 3, 2, 1, 152);

	ITensor *inputTensors153[] = {l152->getOutput(0), l116->getOutput(0)};
	auto cat153 = network->addConcatenation(inputTensors153, 2);

	auto l154 = ConvBnLeaky(network, weightMap, *cat153->getOutput(0), 512, 1, 1, 0, 154);
	auto l155 = ConvBnLeaky(network, weightMap, *l154->getOutput(0), 1024, 3, 1, 1, 155);
	auto l156 = ConvBnLeaky(network, weightMap, *l155->getOutput(0), 512, 1, 1, 0, 156);
	auto l157 = ConvBnLeaky(network, weightMap, *l156->getOutput(0), 1024, 3, 1, 1, 157);
	auto l158 = ConvBnLeaky(network, weightMap, *l157->getOutput(0), 512, 1, 1, 0, 158);
	auto l159 = ConvBnLeaky(network, weightMap, *l158->getOutput(0), 1024, 3, 1, 1, 159);
	IConvolutionLayer *conv160 = network->addConvolutionNd(*l159->getOutput(0), 3 * (YoloParam::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.160.Conv2d.weight"], weightMap["module_list.160.Conv2d.bias"]);
	assert(conv160);
	// 161 is yolo layer

	auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
	const PluginFieldCollection *pluginData = creator->getFieldNames();
	IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
	ITensor *inputTensors_yolo[] = {conv138->getOutput(0), conv149->getOutput(0), conv160->getOutput(0)};
	auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

	yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	//std::cout << "set name out" << std::endl;
	network->markOutput(*yolo->getOutput(0));

	//�жϵ�ǰ�Կ��Ƿ�֧���ض����㾫��
	if ((dtype == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()) || (dtype == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
	{
		std::cout << "Platform doesn't support this precision." << std::endl;
		assert(0);
	}

	// Build engine

	config->setAvgTimingIterations(1);
	config->setMinTimingIterations(1);
	config->setMaxWorkspaceSize(32 * (1 << 20)); //1 << 30 1G   1<<20 1M
	//config->setMaxWorkspaceSize(512 * (1 << 20));//1 << 30 1G   1<<20 1M
	config->setFlag(BuilderFlag::kDEBUG);

	builder->setMaxBatchSize(BATCH_SIZE);

	// Calibrator life time needs to last until after the engine is built.
	std::unique_ptr<IInt8Calibrator> calibrator;

	if (dtype == DataType::kINT8)
	{
		config->setFlag(BuilderFlag::kINT8);
		calibrator.reset(new Int8EntropyCalibrator(BATCH_SIZE, CALIB_IMAGEDIR, CALIBTABLE_PATH, INPUT_H, INPUT_W, INPUT_C, INPUT_BLOB_NAME));
		assert((&calibrator != nullptr) && "Invalid calibrator for INT8 precision");
		config->setInt8Calibrator(calibrator.get());
	}
	else if (dtype == DataType::kHALF)
	{
		config->setFlag(BuilderFlag::kFP16);
	}
	ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "build out" << std::endl;

	// Don't need the network any more
	network->destroy();

	// Release host memory
	for (auto &mem : weightMap)
	{
		free((void *)(mem.second.values));
	}

	return engine;
}

void CreateTRTEngine(const std::string wtspath, const std::string enginepath, DataType dtype)
{

	IHostMemory *modelStream{nullptr};

	// Create builder
	IBuilder *builder = createInferBuilder(gLogger);
	//builder->setMaxBatchSize(maxBatchSize);
	IBuilderConfig *config = builder->createBuilderConfig();

	// Create model to populate the network, then set the outputs and create an engine
	ICudaEngine *engine = createEngine(builder, config, dtype, wtspath);
	assert(engine != nullptr);

	// Serialize the engine
	modelStream = engine->serialize();

	// Close everything down
	engine->destroy();
	builder->destroy();
	assert(modelStream != nullptr);

	std::ofstream p(enginepath, std::ios::binary);
	if (!p)
	{
		std::cerr << "could not open plan output file" << std::endl;
	}
	p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
	std::cout << "Create engine done!" << std::endl;
	modelStream->destroy();
}

int main()
{
	const std::string wtspath = "../model/yolov4-coco/yolov4-416.wts";
	const std::string enginepath = "../Export/model/yolov4-int8-b128.engine";
	DataType dtype = DataType::kINT8;

	CreateTRTEngine(wtspath, enginepath, dtype);

	return 0;
}
