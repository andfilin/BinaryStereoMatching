//============================================================================
// Name        : BinaryStereoMatching.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#ifndef IOSTREAM
#define IOSTREAM
#include <iostream>
#endif

#ifndef OPENCV_HPP
#define OPENCV_HPP
#include <opencv2/opencv.hpp>
#endif




#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>
#include <thread>

#include "OpencvHelper.h"

using namespace cv;

//const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Classroom1-perfect\\im0.png";
//const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Classroom1-perfect\\im1.png";

const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Monopoly\\view1.png";
const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Monopoly\\view5.png";

//const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\cones\\im2.png";
//const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\cones\\im6.png";
const std::string PATH_RESULT = "D:\\studium\\master\\bildverstehen\\imageJ\\have.png";

const int DESCRIPTORBITS = 4096;
const int ELEMS_PER_DESCRIPTOR = DESCRIPTORBITS / 8 / sizeof(int64_t);
const int WINDOWSIZE = 27;
const int GAUSS_SIGMA = 4;

struct brief {
	uint64_t elems[ELEMS_PER_DESCRIPTOR];
};


/*
 * Counts set bits in 64-bit ints using SWAR-Algorithm.
 * See: https://www.playingwithpointers.com/blog/swar.html
 * */
inline int popcount_swar_int64(uint64_t i){
	i = i - ((i >> 1) & 0x5555555555555555);
	i = (i & 0x3333333333333333) + ( (i >> 2) & 0x3333333333333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56;
}

/*
 * Allocate space for an array of descriptors
 * */
brief* allocateDesctriptorArray(int descriptorsCount, int elemsPerDescriptor){
	brief* descriptors = (brief*) malloc(sizeof(brief) * descriptorsCount);
	/*for (int i = 0; i < descriptorsCount; i++) {
		descriptors[i] = (uint64_t*) malloc(sizeof(uint64_t) * elemsPerDescriptor);
	}*/
	return descriptors;
}




/*
 * calculate BRIEF-Descriptor (uint64-array) for every pixel in this row,
 * excluding the border where window does not fit.
 * */
inline void brief_row(uint8_t* pixelrow, int rowWidth, int borderWidth, int* samplepoints, brief* resultBuffer){
	int disparityWidth = rowWidth - borderWidth * 2;

	int descriptors_index = 0;
	// iterate pixels in the row,
	// ignoring border left and right where window does not fit
	for(int col = borderWidth; col < rowWidth - borderWidth; col++){
		// get pointer to current pixel
		uint8_t* currentPtr = pixelrow + col;
		// get buffer where to write this pixels descriptor
		uint64_t* descriptor = resultBuffer[descriptors_index++].elems;

		int sampleIndex = 0;
		// calculate descriptor == array of int64's.
		for(int int64Index = 0; int64Index < ELEMS_PER_DESCRIPTOR; int64Index++){
			uint64_t currentInt64 = 0;
			// each bit is the result of one comparison between two pixels.
			for(uint bitIndex = 0; bitIndex < sizeof(uint64_t) * 8; bitIndex++ ){

				int x1 = samplepoints[sampleIndex++];
				int y1 = samplepoints[sampleIndex++];

				int x2 = samplepoints[sampleIndex++];
				int y2 = samplepoints[sampleIndex++];

				//get first pixel
				uint8_t* pixelA_index = currentPtr + x1;	// apply x
				pixelA_index += rowWidth * y1;	// apply y

				//get second pixel
				uint8_t* pixelB_index = currentPtr + x2;	// apply x
				pixelB_index += rowWidth * y2;	// apply y

				// compare pixels and get bit
				uint64_t cmp = *pixelA_index > *pixelB_index;
				currentInt64 += cmp << bitIndex;
			}
			descriptor[int64Index] = currentInt64;
			//descriptor[int64Index] = currentInt64;
		}
	}
}

 void freeDescriptors(uint64_t** descriptors, int length){
	int ints64_per_descriptor = (DESCRIPTORBITS / 8) / sizeof(uint64_t);
	for(int i = 0; i < length; i++){
		uint64_t* descriptor = descriptors[i];
		free(descriptor);
	}
	free(descriptors);
}

/*
 * For each descriptor in left-array,
 * finds distance to descriptor in right-array with minimal hamming-distance.
 * Returns: int16-array of distances, size is equal to inputarrays.
 * */
inline void compareBriefRows(brief* descriptors_left, brief* descriptors_right, int length, uint16_t* targetBuffer){

	// iterate left descriptors
	for(int indexLeft = 0; indexLeft < length; indexLeft++){
		// initially, set minimal distance to maximum possible
		uint8_t minDisparity = UINT8_MAX;
		uint16_t minHamming = DESCRIPTORBITS;
		// get current left descriptor
		uint64_t* descriptor_left = descriptors_left[indexLeft].elems;
		// iterate right descriptors, starting with an index equal to current leftindex
		for(int indexRight = 0; indexRight < indexLeft + 1; indexRight++){
			// get current right descriptor
			uint64_t* descriptor_right = descriptors_right[indexRight].elems;
			// calculate hammingdistance between both descriptors, by adding up hamming distances of individual uint64_t-pairs
			int16_t currentHammingDistance = 0;
			for(int descIndex = 0; descIndex < ELEMS_PER_DESCRIPTOR; descIndex++){
				uint64_t descriptorElem_left = descriptor_left[descIndex];
				uint64_t descriptorElem_right = descriptor_right[descIndex];

				uint64_t _xor = descriptorElem_left ^ descriptorElem_right;
				uint8_t hamming = __builtin_popcountll(_xor);
				currentHammingDistance += hamming;
				//currentHammingDistance += popcount_swar_int64(_xor);
			}

			// check if current distance is new minimum
			if(currentHammingDistance < minHamming){
				minHamming = currentHammingDistance;
				// new best pixeldistance is difference of col-indices
				uint16_t diff = indexLeft > indexRight ? indexLeft - indexRight : indexRight - indexLeft;
				//diff *= diff*0.5;
				//minDisparity = diff > UINT8_MAX? UINT8_MAX : diff;
				minDisparity = diff > UINT8_MAX? 0 : UINT8_MAX - diff;
			}


		}
		targetBuffer[indexLeft] = UINT8_MAX - minDisparity;
	}
}



/*
	 * returns probabilities for each possible 1D-distance in kernelwindow from center,
	 * from normalized gauss-distribution.
	 * */
double* gaussDist(int windowsize, double sigma){
	int max_x = (windowsize - 1) / 2;
	double* probabilities = (double*) malloc(windowsize * sizeof(double));
	// calculate gauss for each x between -max_x, max_x
	for(int x = -max_x; x <= max_x; x++){
		double e_pot = (-x*x)/(2.0*sigma*sigma);
		double y = (1.0 / (sqrt(2*M_PI) * sigma));
		y *= exp(e_pot);
		probabilities[x + max_x] = y;
	}

	// normalize, so sum is 1
	double sum = 0;
	for(int i = 0; i < windowsize; i++){
		sum += probabilities[i];
	}
	double normFactor = 1.0 / sum;
	std::cout << "\normfactor: " << normFactor << "\n";
	for(int i = 0; i < windowsize; i++){
		probabilities[i] *= normFactor;
		//std::cout << probabilities[i] << "\t";
	}
	//std::cout << "\n";
	return probabilities;
}

/*
	 * Get <numberPoints> random points in Kernel of size <windowSize>.
	 * returned Array contains both x-coords (uneven index) and y-coords (even index).
	 * Coords randomly generated are weighted by normalized gauss-distribution.
	 * */
int* sampleFilterNeighbours(int numberPoints, int windowsize, int sigma) {
	srand(42);
	int* coords = (int*) malloc(sizeof(int) * numberPoints*2);
	double* weights = gaussDist(windowsize, sigma);

	int maxVal = (windowsize - 1) / 2;
	// fill coordsarray
	for(int n = 0; n < numberPoints * 2; n++){
		// get a random coord between -maxVal, maxVal using weights
		int coord;
		double random = rand() / (RAND_MAX + 1.);
		for(int i = 0; i < windowsize; i++) {
			random -= weights[i];
			if(random <= 0) {
				coord = i - maxVal;
				break;
			}
		}
		coords[n] = coord;
		//std::cout << coord << "\t";
	}

	std::cout << "numCoords" << numberPoints*2 << "\n";
	fflush(stdout);
	return coords;
}

/*
 * For each pixel in an image, calculate BRIEF-Descriptor of length <DESCRIPTORBITS>.
 * Each Descriptor is split up in an array of int64_t (8 Bytes).
 * Returns 2D-Array: for each pixel, an int64_t-array.
 * The border of the inputimage is ignored where window does not fit.
 *
 * Result: malloc fails to allocate space at ~2GB of RAM allocated.
 * 		-> calculate Descriptor per Rows instead!
 *
 * */
uint64_t** calculateDescriptors(Mat image, int windowsize, int* pointsamples){
	int width = image.cols;
	int height = image.rows;
	int pixelCount = width * height;

	int borderSize = (windowsize - 1) / 2;

	int descriptorsWidth = width - 2 * borderSize;
	int descriptorsHeight = height - 2 * borderSize;
	int descriptorsCount = descriptorsWidth * descriptorsHeight;

	// get pointer to pixels
	//uint8_t* pixels = image.ptr<uint8_t>(0);
	//int pixelIndex = 0;

	// number of uint64_t's we need to store one descriptor
	int ints64_per_descriptor = (DESCRIPTORBITS / 8) / sizeof(uint64_t);

	// allocate space for one arraypointer per pixel
	uint64_t** descriptors = (uint64_t**) malloc(sizeof(uint64*) * descriptorsCount);
	int descriptors_index = 0;

	// for each pixel(igoring border where window does not fit),
	// calculate its descriptor(as uint64_t-array)
	for(int row = 0; row < height; row++){

		// ignore border vertical
		if(row < borderSize || row > (height - borderSize)) {
			continue;
		}
		// get rowpointer
		uint8_t* rowptr = image.ptr<uint8_t>(row);
		for(int col = 0; col < width; col++) {

			// ignore border horizontal
			if(col < borderSize || col > (width - borderSize)) {
				continue;
			}

			// get pointer to current pixel
			uint8_t* currentPtr = rowptr + col;

			// allocate space for currentpixels descriptor;
			uint64_t* descriptor = (uint64_t*) malloc(sizeof(uint64_t) * ints64_per_descriptor);
			descriptors[descriptors_index++] = descriptor;
			int sampleIndex = 0;

			// calculate descriptor == array of int64's.
			for(int int64Index = 0; int64Index < ints64_per_descriptor; int64Index++){
				uint64_t currentInt64 = 0;
				// each bit is the result of one comparison between two pixels.
				for(uint bitIndex = 0; bitIndex < sizeof(uint64_t) * 8; bitIndex++ ){

					int x1 = pointsamples[sampleIndex++];

					int y1 = pointsamples[sampleIndex++];

					int x2 = pointsamples[sampleIndex++];

					int y2 = pointsamples[sampleIndex++];



					//get first pixel
					uint8_t* pixelA_index = currentPtr + x1;	// apply x

					pixelA_index += width * y1;	// apply y


					//get second pixel
					uint8_t* pixelB_index = currentPtr + x2;	// apply x

					pixelB_index += width * y2;	// apply y


					// compare pixels and get bit
					uint64_t cmp = *pixelA_index > *pixelB_index;
					currentInt64 += cmp << bitIndex;
				}
				descriptor[int64Index] = currentInt64;
			}
		}
	}
	return descriptors;
}




/*
 * takes two images, returns disparitymap.
 * calculates and compares brief-descriptors per row
 *
 * */
uint16_t* BinaryStereoMatching_Threaded(Mat imageLeft, Mat imageRight, int* samplepoints, int windowsize){

	clock_t begin_row, end_row;
	float z_row;

	// ----------------------------------
	// 1. Initialization
	// initialize dimensions and allocate space

	// Get Dimensions
	// assume both images have identical dimensions
	int width = imageLeft.cols;
	int height = imageLeft.rows;
	int pixelsLength = width * height;

	// we will ignore the border of the image where the window does not fit
	int bordersize = (windowsize - 1) / 2;
	int disparitymapWidth = width - 2 * bordersize;
	int disparitymapHeight = height - 2 * bordersize;
	int disparitymapLength = disparitymapWidth * disparitymapHeight;

	// Allocate space to hold a row of descriptors at once, for both images
	brief* leftrow_descriptors =  (brief*) malloc(sizeof(brief) * disparitymapWidth);
	brief* rightrow_descriptors = (brief*) malloc(sizeof(brief) * disparitymapWidth);

	// allocate space for resultarray holding the disparities
	uint16_t* disparitymap = (uint16_t*) malloc(sizeof(uint16_t) * disparitymapLength);
	uint16_t* disparitymaprow = disparitymap;

	// ----------------------------------
	// 2. Iteration
	// calculate disparities per row

	// iterate rows in left image
	for(int row = 0; row < height; row++){

		begin_row = clock();

		// ignore border vertical
		if(row < bordersize || row > (height - bordersize - 1)) {
			continue;
		}

		clock_t begin_desc = clock();
		// get pointers to row in both images
		uint8_t* rowptr_left = imageLeft.ptr<uint8_t>(row);
		uint8_t* rowptr_right = imageRight.ptr<uint8_t>(row);
		// calculate the rows briefdescriptors, write them to corresponding buffer
		brief_row(rowptr_left, width, bordersize, samplepoints, leftrow_descriptors);
		brief_row(rowptr_right, width, bordersize, samplepoints, rightrow_descriptors);

			clock_t begin_compareRows = clock();
		// find disparities by comparing the rows, write result to buffer
		compareBriefRows(leftrow_descriptors, rightrow_descriptors, disparitymapWidth, disparitymaprow);
		disparitymaprow += disparitymapWidth;

			clock_t begin_free = clock();
		//freeDescriptors(leftrow_descriptors, descriptorsWidth);
		//freeDescriptors(rightrow_descriptors, descriptorsWidth);


		end_row = clock();
		z_row=end_row - begin_row;
		z_row/=CLOCKS_PER_SEC;

		float z_cmp=begin_free - begin_compareRows;
		z_cmp/=CLOCKS_PER_SEC;

		float z_free=end_row - begin_free;
		z_free/=CLOCKS_PER_SEC;

		float z_desc=begin_compareRows - begin_desc;
		z_desc/=CLOCKS_PER_SEC;

		/*std::cout << "\n\nrow: " << row;
		std::cout <<" \ntime row: " << z_row << "s" ;
		std::cout <<" \ntime desc: " << z_desc << "s" ;
		std::cout <<" \ntime cmp: " << z_cmp << "s" ;
		std::cout <<" \ntime free: " << z_free << "s" ;*/

	}
	return disparitymap;
}
uint16_t* BinaryStereoMatching(Mat imageLeft, Mat imageRight, int* samplepoints, int windowsize){

	clock_t begin_row, end_row;
	float z_row;

	// ----------------------------------
	// 1. Initialization
	// initialize dimensions and allocate space

	// Get Dimensions
	// assume both images have identical dimensions
	int width = imageLeft.cols;
	int height = imageLeft.rows;
	int pixelsLength = width * height;

	// we will ignore the border of the image where the window does not fit
	int bordersize = (windowsize - 1) / 2;
	int disparitymapWidth = width - 2 * bordersize;
	int disparitymapHeight = height - 2 * bordersize;
	int disparitymapLength = disparitymapWidth * disparitymapHeight;

	// Allocate space to hold a row of descriptors at once, for both images
	brief* leftrow_descriptors =  (brief*) malloc(sizeof(brief) * disparitymapWidth);
	brief* rightrow_descriptors = (brief*) malloc(sizeof(brief) * disparitymapWidth);

	// allocate space for resultarray holding the disparities
	uint16_t* disparitymap = (uint16_t*) malloc(sizeof(uint16_t) * disparitymapLength);
	uint16_t* disparitymaprow = disparitymap;

	// ----------------------------------
	// 2. Iteration
	// calculate disparities per row

	// iterate rows in left image
	for(int row = 0; row < height; row++){

		begin_row = clock();

		// ignore border vertical
		if(row < bordersize || row > (height - bordersize - 1)) {
			continue;
		}

		clock_t begin_desc = clock();
		// get pointers to row in both images
		uint8_t* rowptr_left = imageLeft.ptr<uint8_t>(row);
		uint8_t* rowptr_right = imageRight.ptr<uint8_t>(row);
		// calculate the rows briefdescriptors, write them to corresponding buffer
		brief_row(rowptr_left, width, bordersize, samplepoints, leftrow_descriptors);
		brief_row(rowptr_right, width, bordersize, samplepoints, rightrow_descriptors);

			clock_t begin_compareRows = clock();
		// find disparities by comparing the rows, write result to buffer
		compareBriefRows(leftrow_descriptors, rightrow_descriptors, disparitymapWidth, disparitymaprow);
		disparitymaprow += disparitymapWidth;

			clock_t begin_free = clock();
		//freeDescriptors(leftrow_descriptors, descriptorsWidth);
		//freeDescriptors(rightrow_descriptors, descriptorsWidth);


		end_row = clock();
		z_row=end_row - begin_row;
		z_row/=CLOCKS_PER_SEC;

		float z_cmp=begin_free - begin_compareRows;
		z_cmp/=CLOCKS_PER_SEC;

		float z_free=end_row - begin_free;
		z_free/=CLOCKS_PER_SEC;

		float z_desc=begin_compareRows - begin_desc;
		z_desc/=CLOCKS_PER_SEC;

		/*std::cout << "\n\nrow: " << row;
		std::cout <<" \ntime row: " << z_row << "s" ;
		std::cout <<" \ntime desc: " << z_desc << "s" ;
		std::cout <<" \ntime cmp: " << z_cmp << "s" ;
		std::cout <<" \ntime free: " << z_free << "s" ;*/

	}
	return disparitymap;
}





























/*
 * compare time taken by popcount inlinefunction and builtinfunction
 * */
void compare_popcount(int n){
	std::cout << " comparing popcountFunctions\n";
	// generate random data
	uint64_t* data = (uint64_t*) malloc(n * sizeof(uint64_t));
	for(int i = 0; i < n; i++){
		data[i] = rand();
	}

	int* results1 = (int*) malloc(n * sizeof(int));
	int* results2 = (int*) malloc(n * sizeof(int));

	// test builtin function
	long double t0 = time(0);
	for(int i = 0; i < n; i++){
		results1[i] = __builtin_popcount(data[i]);
		//usleep(100);
	}
	long double t1 = time(0);
	long double datetime_diff_ms = difftime(t1, t0) * 1000.;
	std::cout << "builtInFunction: " << datetime_diff_ms << "\n";

	// test inline function
	t0 = time(0);
	for(int i = 0; i < n; i++){
			results2[i] = popcount_swar_int64(data[i]);
	}
	t1 = time(0);
	datetime_diff_ms = difftime(t1, t0) * 1000.;
	std::cout << "inlineFunction: " << datetime_diff_ms << "\n";

	// assert both functions return same results
	int correct = 1;
	for(int i = 0; i < n; i++){
			if(results1[i] != results2[i]){

				correct = 0;
				break;
			}
	}

	if(correct){
		std::cout << "results match\n";
	}
	else {
		std::cout << "resutlts do NOT match\n";
	}




}


int main() {

	clock_t clock_start = clock();

	int* samples = sampleFilterNeighbours(DESCRIPTORBITS * 2, WINDOWSIZE, GAUSS_SIGMA);
	Mat imageL = openImage_grayscale(PATH_IMAGELEFT);
	Mat imageR = openImage_grayscale(PATH_IMAGERIGHT);

	std::cout << "\nimagetype: " << imageL.type() << "\n";
	fflush(stdout);

	uint16_t* distances = BinaryStereoMatching(imageL, imageR, samples, WINDOWSIZE);

	int rWidth = imageL.cols - (WINDOWSIZE - 1);
	int rHeight = imageL.rows - (WINDOWSIZE - 1);
	int rLength = rWidth * rHeight;


	Mat result = Mat(rHeight, rWidth, CV_16UC1, distances);
	//showImage(result, "result", 1000, 1000);

	imwrite( PATH_RESULT, result);




	clock_t clock_end = clock();
	double z = clock_end - clock_start;
	z/=CLOCKS_PER_SEC;
	std::cout << "finished: " << z << "s" << std::endl;
}
