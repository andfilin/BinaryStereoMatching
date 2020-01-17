/*
 * definitions_imports.h
 *
 *  Created on: 28.12.2019
 *      Author: andre
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

/* define possible inputs as enum and pathstrings*/
enum input {CONES = 0, TEDDY = 1, TSUKUBA = 2, VENUS = 3, SHOPVAC = 4, CLASSROOM = 5, MONOPOLY = 6, ENUMLENGTH = 7};
const std::string CONES_LEFT = ".\\inputs\\cones\\im2.png";
const std::string CONES_RIGHT = ".\\inputs\\cones\\im6.png";
const std::string TEDDY_LEFT = ".\\inputs\\teddy\\im2.png";
const std::string TEDDY_RIGHT = ".\\inputs\\teddy\\im6.png";
const std::string TSUKUBA_LEFT = ".\\inputs\\tsukuba\\scene1.row3.col3.ppm";
const std::string TSUKUBA_RIGHT = ".\\inputs\\tsukuba\\scene1.row3.col4.ppm";
const std::string VENUS_LEFT = ".\\inputs\\venus\\im2.ppm";
const std::string VENUS_RIGHT = ".\\inputs\\venus\\im6.ppm";

const std::string SHOPVAC_LEFT = ".\\inputs\\shopvac\\im0.png";
const std::string SHOPVAC_RIGHT = ".\\inputs\\shopvac\\im1.png";
const std::string CLASSROOM_LEFT = ".\\inputs\\classroom\\im0.png";
const std::string CLASSROOM_RIGHT = ".\\inputs\\classroom\\im1.png";
const std::string MONOPOLY_LEFT = ".\\inputs\\monopoly\\view1.png";
const std::string MONOPOLY_RIGHT = ".\\inputs\\monopoly\\view5.png";

/*For each input, 2 Strings == leftImage, rightimage*/
std::string INPUTPATHS[ENUMLENGTH][2] = {
		{CONES_LEFT, CONES_RIGHT},
		{TEDDY_LEFT, TEDDY_RIGHT},
		{TSUKUBA_LEFT, TSUKUBA_RIGHT},
		{VENUS_LEFT, VENUS_RIGHT},
		{SHOPVAC_LEFT, SHOPVAC_RIGHT},
		{CLASSROOM_LEFT, CLASSROOM_RIGHT},
		{MONOPOLY_LEFT, MONOPOLY_RIGHT}
};

const std::string PATH_RESULT = ".\\result.png";
const std::string PATH_RESULT_EQUALIZED = ".result_equalized.png";

//-------------------------
// choosable parameters
//
// inputimages
const int CHOSENINPUT = TEDDY;
// Number of Bits per Descriptor
const int DESCRIPTORBITS = 4096;
// Size of kernelwindow
const int WINDOWSIZE = 27;
// standardeviation of gauss
const int GAUSS_SIGMA = 4;
// Number of Threads to use
const int THREADCOUNT = std::thread::hardware_concurrency();
// Seed for RNG
const int SEED = 17;

//-------------------------
// consts derived from parameters
//
// number of int64_t's needed to store one descriptor
const int ELEMS_PER_DESCRIPTOR = DESCRIPTORBITS / 8 / sizeof(int64_t);
// in kernel, distance from center to border
const int BORDERSIZE = (WINDOWSIZE - 1) / 2;

// define a brief-descriptor as array of int64_t's
struct brief {
	uint64_t elems[ELEMS_PER_DESCRIPTOR];
};



#endif /* DEFINITIONS_H_ */
