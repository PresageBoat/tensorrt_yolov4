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
//TH_PlateIDCfg plateIDCfg;     //ʶ���������Ϣ
//
//
//int InitPlateID()
//{
//    int nRet;
//
//	memset(&plateIDCfg, 0, sizeof(TH_PlateIDCfg));
//
//	//������С���ƿ�������Ϊ��λ����С����Ϊ60���Ƽ�ֵ80
//	plateIDCfg.nMinPlateWidth = 60;	
//
//	//��������ƿ�������Ϊ��λ��������Ϊ400
//	plateIDCfg.nMaxPlateWidth = 400;  
//
//	//���ͼ���ȣ�����Ϊʵ��ͼ����
//	plateIDCfg.nMaxImageWidth = MAX_IMG_WIDTH;     
//
//	//���ͼ��߶ȣ�����Ϊʵ��ͼ��߶�
//	plateIDCfg.nMaxImageHeight = MAX_IMG_HEIGHT;   
//
//	//�Ƿ�ͼ��	0-�� 1-��
//	plateIDCfg.bIsFieldImage = 0;		
//
//	//�Ƿ�ͬһ�����Ķ��ͼ��ֻ���һ�ν��  0-�� 1-��
//	//��Ƶʶ��ģʽ��Ч
//	plateIDCfg.bOutputSingleFrame = 1;
//
//	//�˶�or��ֹͼ��  0-��ֹ  1-�˶�(ֻ��������֡���˶��Ĳ��ֽ���ʶ����ֻ֧��1���������ٶȽϿ�)
//	//��֡ʶ��ʱ��ֵ��ֵΪ0
//	plateIDCfg.bMovingImage = 0;		
//
//	//ͼ�����ݸ�ʽ:ImageFormatRGB��ImageFormatBGR��ImageFormatYUV422��ImageFormatYUV420COMPASS��ImageFormatYUV420
//	//����Ϊʵ�ʵ�ͼ���ʽ
//	plateIDCfg.nImageFormat = ImageFormatBGR; 	
//
//	//DSPƬ���ڴ�
//	plateIDCfg.pFastMemory = mem1;		
//
//	//DSPƬ���ڴ��С
//	plateIDCfg.nFastMemorySize = MEM1SIZE;  
//	
//	//��ͨ�ڴ�
//	plateIDCfg.pMemory = mem2;			
//
//	//��ͨ�ڴ��С
//	plateIDCfg.nMemorySize = MEM2SIZE;                  
//	
//	//��ʼ������ʶ��SDK����ʹ�ø�SDK�Ĺ���ǰ�����ҽ������һ�θú�����pPlateConfig[in]: ����ʶ��SDK������
//    nRet = TH_InitPlateIDSDK(&plateIDCfg);
//
//	if(nRet != TH_ERR_NONE)
//	{
//		printf("plate id init error = %d\n", nRet);
//		return nRet;
//	}
//	//����Ĭ��ʡ�ݣ�������ʡ���ַ����ŶȽϵ�ʱ��ʶ����ο����õ�Ĭ��ʡ�ݣ����һ�������Ƶ��ַ������֧��6��Ĭ��ʡ��
//	nRet = 0;//TH_SetProvinceOrder("��\0", &plateIDCfg);
//
//	//TH_SetProvinceOrder("���ջ�", &plateIDCfg);
//
//	if(nRet != TH_ERR_NONE)
//	{
//		printf("Set Porvince error = %d\n", nRet);
//		return nRet;
//	}
//	
//	//����ʶ����ֵ
//	//nPlateLocate_Th[in]: ȡֵ��Χ��0-9��ͼƬĬ����ֵ��5��	���ڳ��ƶ�λ����ֵ����ԽС��Խ���׶�λ�����ƣ���׼ȷ�ʻ��½���
//	//nOCR_Th[in]:         ȡֵ��Χ��0-9��ͼƬĬ����ֵ��1�� ���ڳ���ʶ����ֵ����ԽС��Խ����ʶ���ƣ���׼ȷ�ʻ��½���
//	//pPlateConfig[in]: ����ʶ��SDK�����á�
//	TH_SetRecogThreshold( 5, 1, &plateIDCfg); 
//
//	//���ö����⳵�Ƶ�ʶ��dFormat[in]:���⳵�����ͣ�pPlateConfig[in]: ����ʶ��SDK�����á�
//	//�������Ի����ƣ���ѡ
//	//TH_SetEnabledPlateFormat(PARAM_INDIVIDUAL_ON, &plateIDCfg);
//
//	//����˫����ƣ���ѡ
//	TH_SetEnabledPlateFormat(PARAM_TWOROWYELLOW_ON, &plateIDCfg);
//
//	//���������侯�ƣ���ѡ
//	TH_SetEnabledPlateFormat(PARAM_ARMPOLICE_ON, &plateIDCfg);
//
//	//����˫����ƣ���ѡ
//	TH_SetEnabledPlateFormat(PARAM_TWOROWARMY_ON, &plateIDCfg);
//
//	//����ũ�ó��ƣ���ѡ
//	//TH_SetEnabledPlateFormat(PARAM_TRACTOR_ON, &plateIDCfg);
//
//	//����ʹ�ݳ��ƣ���ѡ
//	//TH_SetEnabledPlateFormat(PARAM_EMBASSY_ON, &plateIDCfg);
//
//	//����˫���侯���ƣ���ѡ
//	TH_SetEnabledPlateFormat(PARAM_ARMPOLICE2_ON, &plateIDCfg);
//
//	//�������ڳ��ƣ���ѡ
//	//TH_SetEnabledPlateFormat(PARAM_CHANGNEI_ON, &plateIDCfg);
//
//	//�����񺽳��ƣ���ѡ
//	//TH_SetEnabledPlateFormat(PARAM_MINHANG_ON, &plateIDCfg);
//
//	//�������¹ݳ��ƣ���ѡ
//	//TH_SetEnabledPlateFormat(PARAM_CONSULATE_ON, &plateIDCfg);
//
//	//��������Դ���ƣ���ѡ
//	TH_SetEnabledPlateFormat(PARAM_NEWENERGY_ON, &plateIDCfg);
//
//	
//	//���س���ʶ���汾����ʽ�����汾��.���汾��.�����.ƽ̨����
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
//	std::cout << "������ɫʶ��ʱ�䣺"<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
//
//	if (nRet != 0)
//	{
//		printf("car color = %d", nRet);
//		return nRet;
//	}
//	return nRet;
//}