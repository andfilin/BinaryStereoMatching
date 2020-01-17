//============================================================================
// Name        : BinaryStereoMatching.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================



#include "imports.h"
#include "definitions.h"

using namespace cv;

/*
* calculates probability for each value in 1D-Kernelwindow of size <windowsize>
* using gauss-distribution.
*/
double* gaussDist(int windowsize, double sigma, int randSeed){
	srand(randSeed);
	int max_x = (windowsize - 1) / 2;
	double* probabilities = (double*) malloc(windowsize * sizeof(double));
	double sum = 0;
	// calculate gauss for each x between -max_x, max_x
	for(int x = -max_x; x <= max_x; x++){
		double e_pot = (-x*x)/(2.0*sigma*sigma);
		double y = (1.0 / (sqrt(2*M_PI) * sigma));
		y *= exp(e_pot);

		// test: equaldistribution
		//y = 1;


		probabilities[x + max_x] = y;
		sum += y;
	}

	// normalize, so sum is 1
	double normFactor = 1.0 / sum;
	for(int i = 0; i < windowsize; i++){
		probabilities[i] *= normFactor;
	}
	return probabilities;
}

/*
* generate <numberPoints> * 2 random 1D-Coordinates in a quadratic Kernel of size <windowsize>,
* with center of Kernel = (0, 0),
* using gaussdistribution as weights.
*/
int* randomKernelCoords(int numberPoints, int windowsize, int sigma) {
	// allocate space for result
	int* coords = (int*) malloc(sizeof(int) * numberPoints*2);
	// calculate weights via gauss
	double* weights = gaussDist(windowsize, sigma, SEED);

	// fill coordsarray
	int maxVal = (windowsize - 1) / 2;
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
	}
	return coords;
}

/*
 * calculate BRIEF-Descriptor (uint64_t-array) for every pixel in this row,
 * excluding the border where window does not fit.
 * */
inline void brief_row(uint8_t* pixelrow, int rowWidth, int borderWidth, int* kernelCoords, brief* resultBuffer){
	// iterate pixels in the row,
	// ignoring border left and right where window does not fit
	int currentDescriptor_index = 0;
	for(int col = borderWidth; col < rowWidth - borderWidth; col++){
		// get pointer to current pixel
		uint8_t* currentPtr = pixelrow + col;
		// get buffer where to write this pixels descriptor
		uint64_t* descriptorElems = resultBuffer[currentDescriptor_index++].elems;

		int coordIndex = 0;
		// calculate descriptorElems == array of int64's.
		for(int elemIndex = 0; elemIndex < ELEMS_PER_DESCRIPTOR; elemIndex++){
			uint64_t currentElem = 0;
			// each bit is the result of one comparison between two pixels.
			for(uint bitIndex = 0; bitIndex < sizeof(uint64_t) * 8; bitIndex++ ){
				// get coordinates in kernel of neighbours to compare
				int x1 = kernelCoords[coordIndex++];
				int y1 = kernelCoords[coordIndex++];
				int x2 = kernelCoords[coordIndex++];
				int y2 = kernelCoords[coordIndex++];

				//get first pixel
				uint8_t* pixelA_index = currentPtr + x1 + rowWidth * y1;
				//get second pixel
				uint8_t* pixelB_index = currentPtr + x2 + rowWidth * y2;

				// compare pixels and get current Bit
				uint64_t cmp = *pixelA_index > *pixelB_index;
				currentElem += cmp << bitIndex;
			}
			// set current decriptorelem
			descriptorElems[elemIndex] = currentElem;
		}
	}
}


/*
 * For each descriptor in left-array,
 * finds distance to descriptor in right-array with minimal hamming-distance.
 * writes int8-array of distances to targetBuffer.
 * */
inline void compareBriefRows(brief* descriptors_left, brief* descriptors_right, int length, uint8_t* targetBuffer){
	// iterate left descriptors
	for(int indexLeft = 0; indexLeft < length; indexLeft++){
		// get current left descriptor
		uint64_t* descriptor_left = descriptors_left[indexLeft].elems;
		// initially, set minimal distance to maximum possible
		uint8_t minDisparity = UINT8_MAX;		// save disparity in range 0-255
		uint16_t minHamming = DESCRIPTORBITS;

		// iterate right descriptors - only up to current index of indexLeft
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
			}

			// check if current distance is new minimum
			if(currentHammingDistance < minHamming){
				minHamming = currentHammingDistance;
				// new best pixeldistance is difference of col-indices
				//uint16_t diff = indexLeft > indexRight ? indexLeft - indexRight : indexRight - indexLeft;
				uint16_t diff = indexLeft - indexRight;
				minDisparity = diff > UINT8_MAX ? UINT8_MAX : diff;	// set values over 255 to 255
			}
		}
		// set this pixels disparity
		targetBuffer[indexLeft] = minDisparity;
	}
}

/*
 * Makes Disparity-map from two input images.
 * For Calculating BRIEF-Descriptors, uses coordinates in kernelCoords.
 * */
uint8_t* BinaryStereoMatching(Mat imageLeft, Mat imageRight, int* kernelCoords, int windowsize){
	// ----------------------------------
	// 1. Initialization
	// initialize dimensions and allocate space

	// Get Dimensions
	// assume both images have identical dimensions
	int width = imageLeft.cols;
	int height = imageLeft.rows;
	// ignore the border of the image where the window does not fit
	int bordersize = (windowsize - 1) / 2;
	int disparitymapWidth = width - 2 * bordersize;
	int disparitymapHeight = height - 2 * bordersize;
	int disparitymapLength = disparitymapWidth * disparitymapHeight;

	// Allocate space to hold a row of descriptors at once, for both images
	brief* leftrow_descriptors =  (brief*) malloc(sizeof(brief) * disparitymapWidth);
	brief* rightrow_descriptors = (brief*) malloc(sizeof(brief) * disparitymapWidth);
	// Allocate space for resultarray holding the disparities
	uint8_t* disparitymap = (uint8_t*) malloc(sizeof(uint8_t) * disparitymapLength);
	// pointer where to write results (for one row)
	uint8_t* disparitymaprow = disparitymap;

	// ----------------------------------
	// 2. Iteration
	// calculate disparities per row

	// iterate rows in left image - ignoring border
	for(int row = bordersize; row < (height - bordersize); row++){
		// get pointers to row in both images
		uint8_t* rowptr_left = imageLeft.ptr<uint8_t>(row);
		uint8_t* rowptr_right = imageRight.ptr<uint8_t>(row);
		// calculate the rows briefdescriptors, write them to corresponding buffer
		brief_row(rowptr_left, width, bordersize, kernelCoords, leftrow_descriptors);
		brief_row(rowptr_right, width, bordersize, kernelCoords, rightrow_descriptors);
		// find disparities by comparing the rows, write result to buffer
		compareBriefRows(leftrow_descriptors, rightrow_descriptors, disparitymapWidth, disparitymaprow);
		disparitymaprow += disparitymapWidth;
	}
	return disparitymap;
}
/*
 * Same as BinaryStereoMatching(), but using multiple threads.
 * */
uint8_t* BinaryStereoMatching_Multithreaded(Mat imageLeft, Mat imageRight, int* kernelCoords, int windowsize, int threadnumber){
		// ----------------------------------
		// 1. Initialization
		// initialize dimensions and allocate space
		// ----------------------------------

		// Get Dimensions
		// assume both images have identical dimensions
		int width = imageLeft.cols;
		int height = imageLeft.rows;

		// we will ignore the border of the image where the window does not fit
		int bordersize = (windowsize - 1) / 2;
		int disparitymapWidth = width - 2 * bordersize;
		int disparitymapHeight = height - 2 * bordersize;
		int disparitymapLength = disparitymapWidth * disparitymapHeight;

		// pointers to first imagerows ignoring border
		uint8_t* imageLeft_ptr = imageLeft.ptr<uint8_t>(bordersize);
		uint8_t* imageRight_ptr = imageRight.ptr<uint8_t>(bordersize);

		// number of rows per thread
		uint16_t rowsPerThread = disparitymapHeight / threadnumber;
		uint16_t rowsInLastThread = rowsPerThread + disparitymapHeight % threadnumber;

		// allocate space for resultarray holding the disparities
		uint8_t* disparitymap = (uint8_t*) malloc(sizeof(uint8_t) * disparitymapLength);

		// ----------------------------------
		// 2. Threads
		// make multiple Threads calculate some rows of disparity
		// ----------------------------------

		// collect threads in a vector to join them later
		std::vector<std::thread> threads;
		for(int threadIndex = 0; threadIndex < threadnumber; threadIndex++){
			// ----------------------------------
			// 		Threadparameters
			// ----------------------------------

			// number of rows for this thread
			uint16_t rowCount = (threadIndex == threadnumber - 1) ? rowsInLastThread : rowsPerThread;
			// pointer to this threads first pixel in left image
			uint8_t* leftPtr = imageLeft_ptr + threadIndex * rowsPerThread * width;
			// pointer to this threads first pixel in right image
			uint8_t* rightPtr = imageRight_ptr + threadIndex * rowsPerThread * width;
			// pointer to where this thread can write his first value
			uint8_t* disparitymapPtr = disparitymap + threadIndex * rowsPerThread * disparitymapWidth;

			// ----------------------------------
			// 		start thread
			// ----------------------------------
			std::thread t([&](int rowCount, uint8_t* leftPtr, uint8_t* rightPtr, uint8_t* dispPtr){

				// pointer to current row of disparitymap
				uint8_t* disparitymaprow = dispPtr;

				// Allocate space to hold a row of descriptors at once, for both images
				brief* leftrow_descriptors =  (brief*) malloc(sizeof(brief) * disparitymapWidth);
				brief* rightrow_descriptors = (brief*) malloc(sizeof(brief) * disparitymapWidth);

				// iterate every row this thread has been assigned
				for(int row = 0; row < rowCount; row++){
					// get pointers to row in both images
					uint8_t* rowptr_left = leftPtr + row*width;
					uint8_t* rowptr_right = rightPtr + row*width;

					// calculate the rows briefdescriptors, write them to corresponding buffer
					brief_row(rowptr_left, width, bordersize, kernelCoords, leftrow_descriptors);
					brief_row(rowptr_right, width, bordersize, kernelCoords, rightrow_descriptors);

					// find disparities by comparing the rows, write result to buffer
					compareBriefRows(leftrow_descriptors, rightrow_descriptors, disparitymapWidth, disparitymaprow);
					disparitymaprow += disparitymapWidth;

				}
			},rowCount, leftPtr, rightPtr, disparitymapPtr);

			// add thread to vector
			threads.push_back(std::move(t));
		}
		// join Threads
		for(std::thread & t : threads){
			t.join();
		}

		return disparitymap;

}

Mat openImage_grayscale(std::string path){
	Mat image = imread(path, IMREAD_GRAYSCALE);
	if(image.empty()){
		std::cout << "reading failed\n";
		throw "reading failed";
	}
	return image;
}

Mat addBorder_Grayscale(cv::Mat image, int bordersize){
	Mat result = image;
	int top = bordersize, bottom = bordersize, left = bordersize, right = bordersize;
	copyMakeBorder(image, result, top,  bottom,  left,  right, BORDER_REPLICATE);
	return result;
}


int main(int argc, char** argv) {
	// parameters - if none given as arguments, use defaultvalues
	std::string leftImage_path = INPUTPATHS[CHOSENINPUT][0];
	std::string rightImage_path = INPUTPATHS[CHOSENINPUT][1];
	std::string result_path = PATH_RESULT;
	std::string result_equalized_path = PATH_RESULT_EQUALIZED;
	if(argc == 4){
		leftImage_path = argv[1];
		rightImage_path = argv[2];
		result_path = argv[3];
		std::cout << "using commandline arguments\n ";
	}
	//std::string leftImage_path = INPUTPATHS[CHOSENINPUT][0];

	// print number of threads
	std::cout << "Number of Threads used: " << THREADCOUNT << "\n";
	fflush(stdout);


	// measure runtime
	clock_t clock_start = clock();
	// get kernelCoords
	int* samples = randomKernelCoords(DESCRIPTORBITS * 2, WINDOWSIZE, GAUSS_SIGMA);
	// open Images as Grayscale
	Mat imageL = openImage_grayscale(leftImage_path);
	Mat imageR = openImage_grayscale(rightImage_path);
	// Expand borders of images
	Mat imageL_expanded = addBorder_Grayscale(imageL, BORDERSIZE);
	Mat imageR_expanded = addBorder_Grayscale(imageR, BORDERSIZE);
	// For each pixel, get pixeldisparity from both images
	uint8_t* distances;
	if(THREADCOUNT == 1){
		distances = BinaryStereoMatching(imageL_expanded, imageR_expanded, samples, WINDOWSIZE);
	} else {
		distances = BinaryStereoMatching_Multithreaded(imageL_expanded, imageR_expanded, samples, WINDOWSIZE, THREADCOUNT);
	}
	// make Mat from disparity
	int rWidth = imageL_expanded.cols - (WINDOWSIZE - 1);
	int rHeight = imageR_expanded.rows - (WINDOWSIZE - 1);
	Mat result = Mat(rHeight, rWidth, CV_8UC1, distances);

	// save resultimage
	imwrite(result_path, result);

	// equalize historgram of result and save that too
	equalizeHist(result, result);
	imwrite(result_equalized_path, result);



	// print elapsed time
	clock_t clock_end = clock();
	double z = clock_end - clock_start;
	z/=CLOCKS_PER_SEC;
	std::cout << "finished: " << z << "s" << std::endl;
}
