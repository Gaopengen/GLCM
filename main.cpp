#include <iostream>
#include <Eigen/Dense>//注意Eigen的相关库必须在opencv库的前边
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "GLCM.h"

using std::cout;
using std::endl;

int main() {
    cv::Mat a;
    GLCM glcm;
    double k = 0.0;
    GLCMfeatures glcMfeatures;

    a = cv::imread("1.png");
//    cv::Mat b = cv::Mat::ones(a.rows, a.cols, CV_8UC1);
//    cv::Mat out = cv::Mat::zeros(glcm.my_graylevel,glcm.my_graylevel,CV_8UC1);
    cv::cvtColor(a,a,cv::COLOR_BGR2GRAY);

    //construct test matrix
//    Eigen::Matrix<char, 4, 4> test;
//    test << 0,0,1,1,
//            0,0,1,1,
//            0,2,2,2,
//            2,2,3,3;
//    cv::Mat m;
//    cv::eigen2cv(test, m);
//    k = glcm.calGLCM(m);//test for created matrix

    k = glcm.calGLCM(a);
//    k = glcm.calGLCM(b);
    cout << "k is: " << k << endl;
    cv::imshow("hello", a);
    cv::waitKey(0);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
