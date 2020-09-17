#include "ds_image.h"
#include <experimental/filesystem>

DsImage::DsImage() :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0),
    m_ScalingFactor(0.0),
    m_RNG(cv::RNG(unsigned(std::time(0))))
{
}

DsImage::DsImage(const std::string& path, const int& inputH, const int& inputW) :
    m_Height(0),
    m_Width(0),
    m_XOffset(0),
    m_YOffset(0),
    m_ScalingFactor(0.0),
    m_RNG(cv::RNG(unsigned(std::time(0))))
{
	m_OrigImage = cv::imread(path, cv::IMREAD_COLOR);
	if (m_OrigImage.channels()>3)
	{
		cvtColor(m_OrigImage, m_OrigImage, CV_BGRA2BGR);
	}

    if (!m_OrigImage.data || m_OrigImage.cols <= 0 || m_OrigImage.rows <= 0)
    {
        std::cout << "Unable to open image : " << path << std::endl;
        assert(0);
    }

    if (m_OrigImage.channels() != 3)
    {
		std::cout << "input image channels : " << m_OrigImage.channels() << std::endl;

        std::cout << "Non RGB images are not supported : " << path << std::endl;
        assert(0);
    }
	//
	int w, h, x, y;
	float r_w = inputH / (m_OrigImage.cols*1.0);
	float r_h = inputW / (m_OrigImage.rows*1.0);
	if (r_h > r_w) {
		w = inputW;
		h = r_w * m_OrigImage.rows;
		x = 0;
		y = (inputH - h) / 2;
	}
	else {
		w = r_h * m_OrigImage.cols;
		h = inputH;
		x = (inputW - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(m_OrigImage, re, re.size(), 0, 0, cv::INTER_CUBIC);
	cv::Mat out(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	out.convertTo(m_LetterboxImage, CV_32FC3,1/255.0);
	//m_LetterboxImage = m_LetterboxImage / 255.0;
	//imwrite("D:/mbjc/YOLODLL/configs/process_img_3.jpg", m_LetterboxImage);
	//for (int i = 0; i < inputH * inputW*m_LetterboxImage.channels(); i++) {
	//	std::cout <<(float) m_LetterboxImage.data[i] << std::endl;
	//}


    //m_Height = m_OrigImage.rows;
    //m_Width = m_OrigImage.cols;
    //// resize the DsImage with scale
    //float dim = std::max(m_Height, m_Width);
    //int resizeH = ((m_Height / dim) * inputH);
    //int resizeW = ((m_Width / dim) * inputW);
    //m_ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(m_Height);
    //// Additional checks for images with non even dims
    //if ((inputW - resizeW) % 2) resizeW--;
    //if ((inputH - resizeH) % 2) resizeH--;
    //assert((inputW - resizeW) % 2 == 0);
    //assert((inputH - resizeH) % 2 == 0);
    //m_XOffset = (inputW - resizeW) / 2;
    //m_YOffset = (inputH - resizeH) / 2;
    //assert(2 * m_XOffset + resizeW == inputW);
    //assert(2 * m_YOffset + resizeH == inputH);

    //// resizing
    //cv::resize(m_OrigImage, m_LetterboxImage, cv::Size(inputW, inputH), 0, 0, cv::INTER_CUBIC);
    //// converting to RGB
    //cv::cvtColor(m_LetterboxImage, m_LetterboxImage,cv::COLOR_BGR2RGB);
}