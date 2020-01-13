/*
 * definitions_imports.h
 *
 *  Created on: 28.12.2019
 *      Author: andre
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

//const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Classroom1-perfect\\im0.png";
//const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Classroom1-perfect\\im1.png";
//const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Classroom1-perfect\\im0_mini.png";
//const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Classroom1-perfect\\im1_mini.png";

//const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Monopoly\\view1.png";
//const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\Monopoly\\view5.png";

const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\cones\\im2.png";
const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\cones\\im6.png";

//const std::string PATH_IMAGELEFT = ".\\im2.png";
//const std::string PATH_IMAGERIGHT = ".\\im6.png";

//const std::string PATH_IMAGELEFT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\tsukuba\\scene1.row3.col1.ppm";
//const std::string PATH_IMAGERIGHT = "D:\\studium\\master\\bildverstehen\\Implementierung\\daten\\tsukuba\\scene1.row3.col5.ppm";


const std::string PATH_RESULT = "D:\\studium\\master\\bildverstehen\\imageJ\\result.png";
const std::string PATH_RESULT2 = "D:\\studium\\master\\bildverstehen\\imageJ\\result_equalized.png";
//const std::string PATH_RESULT = ".\\result.png";




//-------------------------
// choosable parameters
//
// Number of Bits per Descriptor
const int DESCRIPTORBITS = 4096;
//const int DESCRIPTORBITS = 8192;
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
