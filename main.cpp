#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "GLCM.h"


int main() {
    cv::Mat a;
    GLCM glcm;
    double k = 0.0;
    GLCMfeatures glcMfeatures;
    a = cv::imread("1.png");
    int row = a.rows;
    int col = a.cols;
//    cv::Mat out = cv::Mat::zeros(glcm.my_graylevel,glcm.my_graylevel,CV_8UC1);
    cv::cvtColor(a,a,cv::COLOR_BGR2GRAY);

    k = glcm.calGLCM(a);
    cv::imshow("hello", a);
    cv::waitKey(0);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
