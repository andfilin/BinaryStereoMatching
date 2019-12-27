/*
 * OpencvHelper.cpp
 *
 *  Created on: 18.12.2019
 *      Author: andre
 */

#ifndef IOSTREAM
#define IOSTREAM
#include <iostream>
#endif

#ifndef OPENCV_HPP
#define OPENCV_HPP
#include <opencv2/opencv.hpp>
#endif


#include "OpencvHelper.h"

using namespace cv;

int test(){
	std::cout << "hello\n";
	return 0;
}

Mat openImage(std::string path){
		Mat image = imread(path);
		if(image.empty()){
			std::cout << "reading failed\n";
			throw "reading failed";
		}
		return image;
}

Mat openImage_grayscale(std::string path){
	Mat image = imread(path, IMREAD_GRAYSCALE);
	if(image.empty()){
		std::cout << "reading failed\n";
		throw "reading failed";
	}
	return image;
}

void showImage(cv::Mat image, std::string title, int windowWidth, int windowHeight){
		namedWindow(title, WINDOW_NORMAL);
		//std::cout << "resizing to: " << windowWidth << " x " << windowHeight;
		resizeWindow(title, windowWidth, windowHeight);
		imshow(title, image);
		waitKey(0);
}

uint8_t* matToInts8(Mat image){
	uint8_t* pixels = image.ptr(0);
	return pixels;
}

Mat ints8ToMat(uint8_t* pixels, int width, int height){
	Mat image = Mat(width, height, CV_8UC1, pixels);
	return image;
}



