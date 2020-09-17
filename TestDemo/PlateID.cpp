//#include <windows.h>
//#include <string.h>
//#include "PlateID.h"
//#include <chrono>
//
//#define MAX_IMG_WIDTH   4200
//#define MAX_IMG_HEIGHT  3000
//
//#define MEM1SIZE 0x4000
//#define MEM2SIZE 120000000
//unsigned char mem1[MEM1SIZE];	// 16K
//unsigned char mem2[MEM2SIZE];
//
//TH_PlateIDCfg plateIDCfg;     //识别库配置信息
//
//
//int InitPlateID()
//{
//    int nRet;
//
//	memset(&plateIDCfg, 0, sizeof(TH_PlateIDCfg));
//
//	//检测的最小车牌宽，以像素为单位，最小可设为60，推荐值80
//	plateIDCfg.nMinPlateWidth = 60;	
//
//	//检测的最大车牌宽，以像素为单位，最大可设为400
//	plateIDCfg.nMaxPlateWidth = 400;  
//
//	//最大图像宽度，设置为实际图像宽度
//	plateIDCfg.nMaxImageWidth = MAX_IMG_WIDTH;     
//
//	//最大图像高度，设置为实际图像高度
//	plateIDCfg.nMaxImageHeight = MAX_IMG_HEIGHT;   
//
//	//是否场图像	0-否 1-是
//	plateIDCfg.bIsFieldImage = 0;		
//
//	//是否同一辆车的多幅图像只输出一次结果  0-否 1-是
//	//视频识别模式有效
//	plateIDCfg.bOutputSingleFrame = 1;
//
//	//运动or静止图像  0-静止  1-运动(只对相邻两帧中运动的部分进行识别，且只支持1个车道，速度较快)
//	//单帧识别时该值赋值为0
//	plateIDCfg.bMovingImage = 0;		
//
//	//图像数据格式:ImageFormatRGB、ImageFormatBGR、ImageFormatYUV422、ImageFormatYUV420COMPASS、ImageFormatYUV420
//	//设置为实际的图像格式
//	plateIDCfg.nImageFormat = ImageFormatBGR; 	
//
//	//DSP片内内存
//	plateIDCfg.pFastMemory = mem1;		
//
//	//DSP片内内存大小
//	plateIDCfg.nFastMemorySize = MEM1SIZE;  
//	
//	//普通内存
//	plateIDCfg.pMemory = mem2;			
//
//	//普通内存大小
//	plateIDCfg.nMemorySize = MEM2SIZE;                  
//	
//	//初始化车牌识别SDK，在使用该SDK的功能前必需且仅需调用一次该函数。pPlateConfig[in]: 车牌识别SDK的配置
//    nRet = TH_InitPlateIDSDK(&plateIDCfg);
//
//	if(nRet != TH_ERR_NONE)
//	{
//		printf("plate id init error = %d\n", nRet);
//		return nRet;
//	}
//	//设置默认省份，当车牌省份字符置信度较低时，识别库会参考设置的默认省份，输出一个较相似的字符，最多支持6个默认省份
//	nRet = 0;//TH_SetProvinceOrder("闽\0", &plateIDCfg);
//
//	//TH_SetProvinceOrder("浙苏沪", &plateIDCfg);
//
//	if(nRet != TH_ERR_NONE)
//	{
//		printf("Set Porvince error = %d\n", nRet);
//		return nRet;
//	}
//	
//	//设置识别阈值
//	//nPlateLocate_Th[in]: 取值范围是0-9，图片默认阈值是5。	用于车牌定位，阈值设置越小，越容易定位出车牌，但准确率会下降。
//	//nOCR_Th[in]:         取值范围是0-9，图片默认阈值是1。 用于车牌识别，阈值设置越小，越容易识别车牌，但准确率会下降。
//	//pPlateConfig[in]: 车牌识别SDK的配置。
//	TH_SetRecogThreshold( 5, 1, &plateIDCfg); 
//
//	//设置对特殊车牌的识别，dFormat[in]:特殊车牌类型，pPlateConfig[in]: 车牌识别SDK的配置。
//	//开启个性化车牌，可选
//	//TH_SetEnabledPlateFormat(PARAM_INDIVIDUAL_ON, &plateIDCfg);
//
//	//开启双层黄牌，可选
//	TH_SetEnabledPlateFormat(PARAM_TWOROWYELLOW_ON, &plateIDCfg);
//
//	//开启单层武警牌，可选
//	TH_SetEnabledPlateFormat(PARAM_ARMPOLICE_ON, &plateIDCfg);
//
//	//开启双层军牌，可选
//	TH_SetEnabledPlateFormat(PARAM_TWOROWARMY_ON, &plateIDCfg);
//
//	//开启农用车牌，可选
//	//TH_SetEnabledPlateFormat(PARAM_TRACTOR_ON, &plateIDCfg);
//
//	//开启使馆车牌，可选
//	//TH_SetEnabledPlateFormat(PARAM_EMBASSY_ON, &plateIDCfg);
//
//	//开启双层武警车牌，可选
//	TH_SetEnabledPlateFormat(PARAM_ARMPOLICE2_ON, &plateIDCfg);
//
//	//开启厂内车牌，可选
//	//TH_SetEnabledPlateFormat(PARAM_CHANGNEI_ON, &plateIDCfg);
//
//	//开启民航车牌，可选
//	//TH_SetEnabledPlateFormat(PARAM_MINHANG_ON, &plateIDCfg);
//
//	//开启领事馆车牌，可选
//	//TH_SetEnabledPlateFormat(PARAM_CONSULATE_ON, &plateIDCfg);
//
//	//开启新能源车牌，可选
//	TH_SetEnabledPlateFormat(PARAM_NEWENERGY_ON, &plateIDCfg);
//
//	
//	//返回车牌识别库版本。格式：主版本号.副版本号.编译号.平台类型
//	//version = TH_GetVersion();
//	//printf("%s\n", version);
//    
//	return nRet;
//}
//
//int RecogImage(const Mat &src, TH_PlateIDResult *result)
//{
//	int nRet;
//
//	if (src.empty())
//	{
//		return -1;
//	}
//
//	TH_SetImageFormat(1, 0, 1, &plateIDCfg);
//	TH_RECT rect;
//	rect.top    = 0;
//	rect.left   = 0;
//	rect.bottom = src.rows; 
//	rect.right = src.cols; 
//	
//	int nResultNum = 6;
//
//	nRet = TH_RecogImage((const unsigned char *)(src.data),src.cols, src.rows, result, &nResultNum, &rect, &plateIDCfg);
//	if (nRet!=0)
//	{
//		printf("reg plate id error = %d", nRet);
//		return nRet;
//	}
//	auto start = std::chrono::system_clock::now();
//
//	nRet = TH_EvaluateCarColor((const unsigned char *)(src.data), src.cols, src.rows, result, &nResultNum, &rect, &plateIDCfg);
//	auto end = std::chrono::system_clock::now();
//	std::cout << "车辆颜色识别时间："<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//
//	if (nRet != 0)
//	{
//		printf("car color = %d", nRet);
//		return nRet;
//	}
//	return nRet;
//}