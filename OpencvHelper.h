/*
 * OpencvHelper.h
 *
 *  Created on: 18.12.2019
 *      Author: andre
 */

#ifndef OPENCVHELPER_H_
#define OPENCVHELPER_H_

#ifndef IOSTREAM
#define IOSTREAM
#include <iostream>
#endif

#ifndef OPENCV_HPP
#define OPENCV_HPP
#include <opencv2/opencv.hpp>
#endif

int test();
cv::Mat openImage(std::string path);
cv::Mat openImage_grayscale(std::string path);
void showImage(cv::Mat image, std::string title, int windowWidth, int windowHeight);
char bytesToMat();
uint8_t* matToInts8(cv::Mat image);
cv::Mat ints8ToMat(uint8_t* pixels, int width, int height);



#endif /* OPENCVHELPER_H_ */
