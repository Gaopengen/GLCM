//
// Created by gao on 2020/2/5.
//

#ifndef OPENCAPI_GLCM_H
#define OPENCAPI_GLCM_H

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>

typedef struct _GLCMfeatures{
    double ASM = 0.0;
    double CON = 0.0;
    double IDM = 0.0;
    double ENT = 0.0;
    double COR = 0.0;
    double k = 0.0;
} GLCMfeatures;

class GLCM{
public:
    enum{
        GLCM_HORIZATION = 0,        // 水平
        GLCM_VERTICAL = 1,          // 垂直
        GLCM_ANGLE45 = 2,           // 45度角
        GLCM_ANGLE135 = 3,          // 135度角
        MY_GRAYLEVEL = 256          // 像素灰度级
    };

    GLCM();
    cv::Mat mat_heri = cv::Mat::zeros(MY_GRAYLEVEL,MY_GRAYLEVEL,CV_32FC1);
    cv::Mat mat_ver = cv::Mat::zeros(MY_GRAYLEVEL,MY_GRAYLEVEL,CV_32FC1);
    cv::Mat mat_ang45 = cv::Mat::zeros(MY_GRAYLEVEL,MY_GRAYLEVEL,CV_32FC1);
    cv::Mat mat_ang135 = cv::Mat::zeros(MY_GRAYLEVEL,MY_GRAYLEVEL,CV_32FC1);

public:
    double calGLCM(cv::Mat mat_in, int angle = 0);
    void calFeature(cv::Mat mat_in, GLCMfeatures &features);

private:
    // 计算水平灰度共生矩阵
    void getHorizonGLCM(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight);
    // 计算垂直灰度共生矩阵
    void getVerticalGLCM(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight);
    // 计算 45 度灰度共生矩阵
    void getGLCM45(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight);
    // 计算 135 度灰度共生矩阵
    void getGLCM135(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight);
    //矩阵归一化
    void normalization(cv::Mat &src);

public:
    int my_graylevel = 256;
};

//create a 2 dimension matrix
template <typename T>
void createMatrix(int const row, int const col, T** &m){
    m = new T* [row];
    for(int i = 0; i < row; i++){
        m[i] = new T[col];
    }
}

#endif //OPENCAPI_GLCM_H
