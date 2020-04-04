//
// Created by gao on 2020/2/5.
//

#include "GLCM.h"
#include <math.h>
#include <iostream>

using std::cout;
using std::endl;

//Initialize GLCM matrix
GLCM::GLCM(){}


double GLCM::calGLCM(cv::Mat mat_in, int angle) {

    int rows = mat_in.rows;
    int cols = mat_in.cols;

    GLCMfeatures f_h,f_v,f_45,f_135;

    getHorizonGLCM(mat_in, mat_heri, cols, rows);
    getVerticalGLCM(mat_in, mat_ver, rows, cols);
    getGLCM45(mat_in, mat_ang45, rows, cols);
    getGLCM135(mat_in, mat_ang135, rows, cols);


    std::cout << mat_heri << std::endl;
    std::cout << mat_ver << std::endl;
    std::cout << mat_ang45 << std::endl;
    std::cout << mat_ang135 << std::endl;


    calFeature(mat_heri, f_h);
    calFeature(mat_ver,f_v);
    calFeature(mat_ang45,f_45);
    calFeature(mat_ang135,f_135);

    double k = 0.0;
    k = (f_h.k + f_v.k + f_45.k + f_135.k)/4.0;
    return k;
}


void GLCM::calFeature(cv::Mat mat_in, GLCMfeatures &features) {

    int L = MY_GRAYLEVEL;
    int i,j;

    //ASM
    for(i = 0; i < L; i++)
        for(j = 0; j < L; j++){
            features.ASM += pow(mat_in.at<float>(i,j),2);
        }

    //CON
//    double CON = 0.0;
    for(i = 0; i < L; i++)
        for(j = 0; j < L; j++){
            features.CON += pow(i-j,2)*mat_in.at<float>(i,j);
        }

    //IDM
//    double IDM = 0.0;
    for(i = 0; i < L; i++)
        for(j = 0; j < L; j++){
            features.IDM += mat_in.at<float>(i,j)/(1+pow(i-j,2));
        }

    //ENT
//    double ENT = 0.0;
    for(i = 0; i < L; i++)
        for(j = 0; j < L; j++){
            if(mat_in.at<float>(i,j) > 0.00000003)
                features.ENT += -mat_in.at<float>(i,j) * log2(mat_in.at<float>(i,j));
        }

    //COR
    double ux,uy,sigmax,sigmay;
//    double COR;
    ux = uy = sigmax = sigmay = 0.0;
    for(i = 0; i < L; i++)
        for(j = 0; j < L; j++){
            ux += i * mat_in.at<float>(i,j);
            uy += j * mat_in.at<float>(i,j);
        }

    for(i = 0; i < L; i++)
        for(j = 0; j < L; j++){
            sigmax += pow(i-ux,2)*mat_in.at<float>(i,j);
            sigmay += pow(j-uy,2)*mat_in.at<float>(i,j);
        }

    for(i = 0; i < L; i++)
        for(j = 0; j < L; j++){
            features.COR += i * j * mat_in.at<float>(i,j);
        }
    if(features.COR - ux * uy > 0.00000003)
        features.COR = (features.COR - ux * uy)/(sigmax * sigmay);
    else
        features.COR = 0.0;
    features.k = features.CON + features.ENT - features.ASM - features.IDM - features.COR;
}


void GLCM::getHorizonGLCM(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight) {
    int i,j;
    unsigned char vi, vj;

    for(i = 0; i < imgHeight; i++){
        for(j = 0; j < imgWidth; j++){
            vi = src.at<uchar>(i,j);
            if(j+1 < imgWidth){
                vj = src.at<uchar>(i,j+1);
                dst.at<float>(vi,vj) = dst.at<float>(vi, vj) + 1.0;
            }
            if(j-1 >= 0){
                vj = src.at<uchar>(i,j-1);
                dst.at<float>(vi,vj) = dst.at<float>(vi, vj) + 1.0;
            }
        }
    }

    normalization(dst);
}


void GLCM::getVerticalGLCM(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight) {
    int i,j;
    unsigned char vi, vj;

    for (i = 0; i < imgHeight; i++)
        for(j = 0; j < imgWidth; j++) {
            vi = src.at<uchar>(i, j);
            if(i+1 < imgHeight){
                vj = src.at<uchar>(i+1, j);
                dst.at<float>(vi, vj) = dst.at<float>(vi, vj) + 1.0;
            }
            if(i-1 >= 0){
                vj = src.at<uchar>(i-1,j);
                dst.at<float>(vi, vj) = dst.at<float>(vi, vj) + 1.0;
            }
        }
    normalization(dst);
}

void GLCM::getGLCM45(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight) {
    int i,j;
    unsigned char vi, vj;

    for(i = 0; i < imgHeight; i++)
        for(j = 0; j < imgWidth; j++){
            vi = src.at<uchar>(i,j);
            if(i + 1 < imgHeight && j + 1 < imgWidth){
                vj = src.at<uchar>(i+1, j+1);
                dst.at<float>(vi, vj) += 1.0;
            }
            if(i-1 >= 0 && j-1 >= 0){
                vj = src.at<uchar>(i-1, j-1);
                dst.at<float>(vi, vj) += 1.0;
            }
        }
    normalization(dst);
}

void GLCM::getGLCM135(cv::Mat src, cv::Mat &dst, int imgWidth, int imgHeight) {
    int i,j;
    unsigned char vi, vj;

    for (i = 0; i < imgHeight; i++)
        for(j = 0; j < imgWidth; j++) {
            vi = src.at<uchar>(i, j);
            if(i-1 >= 0 && j+1 < imgWidth){
                vj = src.at<uchar>(i-1, j+1);
                dst.at<float>(vi, vj) += 1.0;
            }
            if(i + 1 < imgHeight && j-1 >=0){
                vj = src.at<uchar>(i+1, j-1);
                dst.at<float>(vi, vj) += 1.0;
            }
        }

    normalization(dst);
}


void GLCM::normalization(cv::Mat &mat){
    float total = 0.0;
    int i,j;
    for(i = 0; i < MY_GRAYLEVEL; i++)
        for(j = 0; j < MY_GRAYLEVEL; j++){
            total += mat.at<float>(i,j);
        }

    for(i = 0; i < MY_GRAYLEVEL; i++)
        for(j = 0; j < MY_GRAYLEVEL; j++){
            mat.at<float>(i,j) = mat.at<float>(i,j)/total;
        }
}