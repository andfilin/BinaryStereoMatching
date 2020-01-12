/*
 * OpencvHelper.cpp
 *
 *  Created on: 18.12.2019
 *      Author: andre
 */

#include "imports.h"
#include "definitions.h"
#include "OpencvHelper.h"

using namespace cv;

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


Mat addBorder_Grayscale(cv::Mat image, int bordersize){
	Mat result = image;
	int top = bordersize, bottom = bordersize, left = bordersize, right = bordersize;
	copyMakeBorder(image, result, top,  bottom,  left,  right, BORDER_REPLICATE);
	return result;

}


