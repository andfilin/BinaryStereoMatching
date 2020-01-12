/*
 * OpencvHelper.h
 *
 *  Created on: 18.12.2019
 *      Author: andre
 */

#ifndef OPENCVHELPER_H_
#define OPENCVHELPER_H_

cv::Mat openImage(std::string path);
cv::Mat openImage_grayscale(std::string path);
void showImage(cv::Mat image, std::string title, int windowWidth, int windowHeight);

cv::Mat addBorder_Grayscale(cv::Mat image, int bordersize);


#endif /* OPENCVHELPER_H_ */
