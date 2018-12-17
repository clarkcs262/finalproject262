// CS262 Final Project
// Authors: Aidan McLaughlin, Bo Liu
//
// This program reads an image from a file, converts it to grayscale, 
// then calculates the derivatives in the X and Y directions. 
// The image and the derivative images are displayed in separate window. 
// If a user clicks in the image, the value of the x and y components of 
// the gradient at the clicked location is printed to the terminal, 
// and a red box is displayed overlayed on the image.
//
// Copyright 2018 Diane H. Theriault, John Magee
//
//

// If you paste this into a new visual studio project, change "stdafx.h" to "pch.h"
//#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/optflow.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// Math library will require -lm to link math on linux/mac
//#include <math.h>
//#include <stdio.h>

using namespace std;
using namespace cv;


//realist class; returns confidence score that the image is an example of realism
/*class realistScore {
	Mat greyVersion = grayImage;

	//face detection

	//gradients for perspective; 

	//realist painting should remain more true to original as it is smoothed

	//face detection

	//if a face is detected AND there is low saturation throughout the image,
	//image more likely to be realist

	//if saturation is above threshold (maybe 70) then image less likely to be realism

	//more likely realism if it has a well defined, accurate shadow -- 
	//pointilist images have more abstract, less pronounced shadows or no shadows at all

};
*/

/*
class pointilist {
	//pointilist image should have many more edges than a realist, so count edges detected

	//also, many pointilist images should be mostly unrecognizable when blurred, so try checking SIFT feature detection
};
*/

//Global variables are evil, but easy
Mat original, image, displayImage;
Mat maskImage;
Mat dX, dY, gradientMagnitude, laplacian;
Mat displayDX, displayDY, displayMagnitude, displayLaplacian;
int outputCounter;
double pointlismP = 0.0;
double realismP = 0.0;
int smoothSlider;
int smoothSliderMax = 10;

//what not to do
void gradient_FiniteDifferences_v1(Mat& image, Mat& dX, Mat& dY);
void gradient_FiniteDifferences_v2(Mat& image, Mat& dX, Mat& dY);
void gradient_Convolution(Mat& image, Mat& dX, Mat& dY);
void convolutionByHand(Mat& image, Mat& coefficients, Mat& result);


//the better way to do it
void gradient_OpenCV_v1(Mat& image, Mat& dX, Mat& dY);
void gradient_OpenCV_v2(Mat& image, Mat& dX, Mat& dY);

//smooth image
void smoothImage(Mat& image, double sigma);

//compute derivatives and gradient magnitude
void gradientSobel(Mat& image, Mat& dX, Mat& dY, Mat& magnitude);

//convert gradient image for display by scaling into the range [0 255]
bool convertGradientImageForDisplay(Mat& input, Mat& output);

//duplicate the grayscale image 3 times to get a 3 channel image so that we
//can then manipulate parts of it to paint them red
bool markImageForDisplay(Mat& gray, Mat& output, Mat& mask);

//read in the image and convert it to grayscale
bool readImageAndConvert(const string& filename, Mat& grayImage);

//the callback for the click
void onClickCallback(int event, int x, int y, int q, void* data);

//a function to smooth the image, based on the smoothing factor chosen with the trackbar
void onTrackbar(int value, void* data);

//a function to count the average saturation in an image
double avgSat(Mat& input, int sat);

//a function to count the average Value in an image
double avgValue(Mat& input, int sat);

//SIFT comparison of two images
void SIFTcompare(Mat& input1, Mat& input2, int blurSize);

int main(int argc, char* argv[])
{
	// initialize cascade classifier to use in face detection [149]
	CascadeClassifier face_cascade;

	// check if there was an error creating the cascade classifier
	if (!face_cascade.load("haarcascade_frontalface_alt.xml"))
	{
		printf("--(!)Error loading\n");
	}

	// create vector for the face rectangle
	vector<Rect> faceRect;

	if (argc < 2)
	{
		cout << "Usage: Lab03 imageName" << endl;
		return 0;
	}
	outputCounter = 1;

	//load in the image and convert it to gray scale
	readImageAndConvert(argv[1], original);

	Mat inputImg;
    Mat originalColor;
    originalColor = imread(argv[1],CV_LOAD_IMAGE_COLOR);
	readImageAndConvert(argv[1], inputImg);
    
    Mat originalFblur;
    originalFblur = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    
	// make a copy of original image to use below in line 145/146 - gray image 
	Mat originalImg = original;

	if (original.empty())
	{
		cout << "Unable to open the image file: " << argv[1] << endl;
		return 0;
	}
	original.copyTo(image);

	// make a gray image to use in detecting faces
	Mat FaceDetectGrayImg;
	//FaceDetectGrayImg.create(argv[1].size(), argv[1].type());

	// now use gray image to detect faces if there are any
	face_cascade.detectMultiScale(inputImg, faceRect, 1.05, 3, 0, Size(20, 20), Size(80, 80));

	
	// check whether or not a face(s) was detected
	if (faceRect.size() != 0)
	{
        double valValue;
        double satValue;
        valValue = 0.0;
        satValue = 0.0;
        satValue = avgSat(originalColor, 70);
        valValue = avgValue(originalColor, 60);
        cout << "satValue: " << satValue << endl;
        if(satValue<0.4 &&valValue <0.5){
            //cout << "satValue: " << satValue << endl;
            realismP += 0.3;
            
            }
        else{
            realismP    += 0.2;
            pointlismP += 0.15;
            //cout << "satValue: " << satValue << endl;
        }
	}
    else{
        double valValue;
        double satValue;
        valValue = 0.0;
        satValue = 0.0;
        satValue = avgSat(originalColor, 90);
        valValue = avgValue(originalColor, 60);
        if(satValue<0.35 && valValue<0.5){
            //cout << "satValue: " << satValue << endl;
            realismP += 0.3;
            pointlismP += 0.1;
        }
        else if(satValue>0.3 &&satValue<0.6&& valValue<0.7){
            realismP    += 0.2;
            pointlismP += 0.15;
            //cout << "satValue: " << satValue << endl;
        }
        else{
            realismP    -=0.05;
            pointlismP += 0.3;
        }
        //cout << "satValue: " << satValue << endl;
        //cout << "valValue: " << valValue << endl;

    }
    //cout << "face: " << faceRect.size() << endl;

    cout << "Realism Score: " << realismP << endl;
    cout << "Pointilism Score: " << pointlismP << endl;
	

	//initialize the versions of the image that we will use for displaying some things
	displayImage.create(image.rows, image.cols, CV_8UC3);
	maskImage = Mat::zeros(image.rows, image.cols, CV_8UC1);
	markImageForDisplay(image, displayImage, maskImage);

	//allocate objects for holding derivatives and gradients
	dX.create(image.rows, image.cols, CV_32F);
	dY.create(image.rows, image.cols, CV_32F);
	gradientMagnitude.create(image.rows, image.cols, CV_32F);
	laplacian.create(image.rows, image.cols, CV_32F);

	//allocate objects for converting gradient images for display
	displayDX.create(dX.rows, dX.cols, CV_8UC1);
	displayDY.create(dY.rows, dY.cols, CV_8UC1);
	displayMagnitude.create(gradientMagnitude.rows, gradientMagnitude.cols, CV_8UC1);
	displayLaplacian.create(laplacian.rows, laplacian.cols, CV_8UC1);

	// Create a new windows
	namedWindow("Image View", 1);

	//attach a mouse click callback to the window
	setMouseCallback("Image View", onClickCallback, NULL);
	//create a slider in the window
	createTrackbar("Smoothing", "Image View", &smoothSlider, smoothSliderMax, onTrackbar);

	//all of the stuff gets computed in the onTrackbar function so that things get recomputed 
	//when you apply different levels of smoothing
	//Here, we call it manually for initialization
	onTrackbar(0, NULL);

	//two more windows for displaying the derivative images
    //namedWindow("originalImg", 1);
	//namedWindow("DX", 1);
	//namedWindow("DY", 1);
	//namedWindow("Gradient Magnitude", 1);
	//namedWindow("Laplacian", 1);

	//display the images and wait for 33 milliseconds in a loop, to 
	//allow us to refresh the displayed 
	while (1)
	{
        imshow(" testing",originalColor);
		//imshow("Image View", displayImage);
		//imshow("DX", displayDX);
		//imshow("DY", displayDY);
		//imshow("Gradient Magnitude", displayMagnitude);
		//imshow("Laplacian", displayLaplacian);
		char key = waitKey(33);
		if (key == 'Q' || key == 'q')
		{
			break;
		}
		if (key == 'S' || key == 's')
		{
			char filename[512];
			//sprintf_s(filename, "outputFile_%03d.png", outputCounter);
			imwrite(filename, displayImage);
			cout << "Image Saved: " << filename << endl;
			outputCounter++;
		}
	}

	return 0;
}

//a function to count the average saturation in an image
double avgSat(Mat& input, int sat)
{   double numofs;
    numofs = 0.0;
    int numPixels;
    
    // new matrix to hold HSV image
    Mat HSV;
    
    // convert RGB image to HSV
    cvtColor(input, HSV, CV_BGR2HSV);

    vector<Mat> hsv_planes;
    split(HSV, hsv_planes);
    Mat h = hsv_planes[0]; // H channel
    Mat s = hsv_planes[1]; // S channel
    Mat v = hsv_planes[2]; // V channel
    
    numPixels = s.total();
    
    
    
    for (int i = 0; i < s.rows; i++)
    {
        for (int j = 0; j < s.cols; j++)
        {
            double value = 0.0;
            value = (int)s.at<uchar>(i, j);
            if (value > sat)
            numofs ++;
        }
    }
    double pre;
    pre  = 0.0;
    pre = numofs / numPixels;
    //cout << "numofs: " << numofs << endl;
    //cout << "numPixels: " << numPixels << endl;
    //cout << "pre: " << pre << endl;
    return pre;
    //convert from RGB to HSV
    //then split into three arrays, H, S, V
    //iterate through S and average this
    
}

double avgValue(Mat& input, int sat){
    double numofs;
    numofs = 0.0;
    int numPixels;
    
    // new matrix to hold HSV image
    Mat HSV;
    
    // convert RGB image to HSV
    cvtColor(input, HSV, CV_BGR2HSV);
    
    vector<Mat> hsv_planes;
    split(HSV, hsv_planes);
    Mat h = hsv_planes[0]; // H channel
    Mat s = hsv_planes[1]; // S channel
    Mat v = hsv_planes[2]; // V channel
    
    numPixels = s.total();
    
    
    
    for (int i = 0; i < s.rows; i++)
    {
        for (int j = 0; j < s.cols; j++)
        {
            double value = 0.0;
            value = (int)v.at<uchar>(i, j);
            if (value > sat)
                numofs ++;
        }
    }
    double pre;
    pre  = 0.0;
    pre = numofs / numPixels;
    //cout << "numofs: " << numofs << endl;
    //cout << "numPixels: " << numPixels << endl;
    //cout << "pre: " << pre << endl;
    return pre;
    //convert from RGB to HSV
    //then split into three arrays, H, S, V
    //iterate through S and average this
    
}

//SIFT comparison of two images
/*
void SIFTcompare(Mat& input1, int blurSize)
{
    //make and display blurred image based on original image
    Mat blurImage;
    GaussianBlur(input1, blurImage, Size(blurSize, blurSize), 0, 0, BORDER_DEFAULT);
    Mat image1, image2;
    readImageAndConvert(input1, image1);
    readImageAndConvert(blurImage, image2);
    Mat grayImage1, grayImage2;
    cvtColor(image1, grayImage1, COLOR_BGR2GRAY);
    cvtColor(image2, grayImage2, COLOR_BGR2GRAY);
    
    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    detector->detect(grayImage1, keypoints1);
    detector->detect(grayImage2, keypoints2);
    
    // http://docs.opencv.org/modules/features2d/doc/common_interfaces_of_descriptor_extractors.html
    Mat descriptors1;
    Mat descriptors2;
    detector->compute(image1, keypoints1, descriptors1);
    detector->compute(image2, keypoints2, descriptors2);
    imshow("blurred window", originalFblur);
}
**/
/*
void detectFaces(Mat& image, CascadeClassifier& face_cascade, vector<Rect>& faces)
{

	// http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html
	// http://docs.opencv.org/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html

	// might not need this later on, use existing gray image 
	Mat frame_gray;
	cvtColor(image, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//imshow("gray image", frame_gray);
	//-- Detect faces
	// in the future, work on rescaling this face-rectangle appropriately
	face_cascade.detectMultiScale(grayImage, faceRect, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
}
*/

void gradient_FiniteDifferences_v1(Mat& image, Mat& dX, Mat& dY)
{
	//in two double for loops, calculate the difference between consecutive elements
	if (image.empty())
	{
		cout << "Error: image is unallocated" << endl;
		return;
	}

	if (dX.empty())
	{
		dX = Mat::zeros(image.rows, image.cols, CV_32FC1);
	}

	if (dY.empty())
	{
		dY = Mat::zeros(image.rows, image.cols, CV_32FC1);
	}

	for (int row = 1; row<image.rows - 1; row++)
	{
		unsigned char* imgPtr = image.ptr<unsigned char>(row);
		unsigned char* imgPtr_above = image.ptr<unsigned char>(row - 1);
		float* dxPtr = dX.ptr<float>(row);
		float* dyPtr = dY.ptr<float>(row);
		for (int col = 1; col<image.cols - 1; col++)
		{
			//This is terrible! It isn't even centered on the pixel that we are working on!
			dxPtr[col] = imgPtr[col + 1] - imgPtr[col];
			dyPtr[col] = imgPtr_above[col] - imgPtr[col];
		}
	}
}

// here is one way of calculating a measure of the derivative with wider spatial support
void gradient_FiniteDifferences_v2(Mat& image, Mat& dX, Mat& dY)
{
	if (image.empty())
	{
		cout << "Error: image is unallocated" << endl;
		return;
	}

	if (dX.empty())
	{
		dX = Mat::zeros(image.rows, image.cols, CV_32FC1);
	}

	if (dY.empty())
	{
		dY = Mat::zeros(image.rows, image.cols, CV_32FC1);
	}

	for (int row = 1; row<image.rows - 1; row++)
	{
		unsigned char* imgPtr = image.ptr<unsigned char>(row);
		unsigned char* imgPtr_above = image.ptr<unsigned char>(row - 1);
		unsigned char* imgPtr_below = image.ptr<unsigned char>(row + 1);
		float* dxPtr = dX.ptr<float>(row);
		float* dyPtr = dY.ptr<float>(row);
		for (int col = 1; col<image.cols - 1; col++)
		{
			//At least this is centered, but good grief. So ugly!
			dxPtr[col] = imgPtr_above[col + 1] + 2 * imgPtr[col + 1] + imgPtr_below[col + 1]
				- imgPtr_above[col - 1] - 2 * imgPtr[col - 1] - imgPtr_below[col - 1];

			dyPtr[col] = imgPtr_above[col - 1] + 2 * imgPtr_above[col] + imgPtr_above[col + 1]
				- imgPtr_below[col - 1] - 2 * imgPtr_below[col] - imgPtr_below[col + 1];
		}
	}
}

void gradient_Convolution(Mat& image, Mat& dX, Mat& dY)
{
	Mat kernelY = Mat::zeros(3, 3, CV_32F);
	kernelY.at<float>(0, 0) = 1;
	kernelY.at<float>(0, 1) = 2;
	kernelY.at<float>(0, 2) = 1;
	kernelY.at<float>(2, 0) = -1;
	kernelY.at<float>(2, 1) = -2;
	kernelY.at<float>(2, 2) = -1;

	Mat kernelX;
	kernelY.copyTo(kernelX);
	transpose(kernelX, kernelX);

	convolutionByHand(image, dX, kernelX);
	convolutionByHand(image, dY, kernelX);

}

//Instead of writing a new function for every set of coefficients we want to use, we
//can put the coefficients in a separate matrix and then do a quadruple for loop
//This process is called "convolution", and the set of coefficients is called a 
//"kernel", "filter", or "mask"
//This is the logic of convolution 
//Note: I didn't actually test that the function works. I just wanted to illustrate
void convolutionByHand(Mat& image, Mat& result, Mat& coefficients)
{
	//assume the size of the coefficients is odd
	int coeffRows = (coefficients.rows - 1) / 2;
	int coeffCols = (coefficients.cols - 1) / 2;

	//change the iteration conditions so that we don't try to access memory incorrectly    
	for (int row = coeffRows; row<image.rows - coeffRows - 1; row++)
	{
		for (int col = coeffCols; col<image.cols - coeffCols - 1; col++)
		{
			//initialize the result
			float* resultPtr = result.ptr<float>(row);
			resultPtr[col] = 0;

			//go through all the elements of the coefficients, and accumulate the result
			for (int dr = -coeffRows; dr<coeffRows; dr++)
			{
				for (int dc = coeffCols; dc<coeffCols; dc++)
				{
					unsigned char* imgPtr = image.ptr<unsigned char>(row + dr);

					//dr+coeffRows so that we don't try to use negative indexes
					float* coefPtr = coefficients.ptr<float>(dr + coeffRows);
					resultPtr[col] += imgPtr[col + dc] * coefPtr[dc + coeffCols];
				}
			}
		}
	}
}


void gradient_OpenCV_v1(Mat& image, Mat& dX, Mat& dY)
{
	//oh, phew. I knew there had to be an easier way
	//
	//http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=sobel#sobel
	//calculate the gradients with OpenCV Sobel() function
	Sobel(image, dX, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(image, dY, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
}

void gradient_OpenCV_v2(Mat& image, Mat& dX, Mat& dY)
{
	// The most general way of computing any linear combinations of elements of your image
	// is with a convolution or filtering operation.
	//
	// The set of coefficients are stored in another matrix, which is referred to as a
	// "kernel", "filter", "operator" or "mask" (a convolution mask, not to be confused with a binary mask)
	//
	//calculate the gradients with more general Convolution function
	//http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=filter2d#filter2d

	//construct the sobel kernels. This is just an example of how to construct a kernel. For using the 
	//Sobel operators to compute derivatives, you should really use the Sobel() function
	Mat kernelY = Mat::zeros(3, 3, CV_32F);
	kernelY.at<float>(0, 0) = 1;
	kernelY.at<float>(0, 1) = 2;
	kernelY.at<float>(0, 2) = 1;
	kernelY.at<float>(2, 0) = -1;
	kernelY.at<float>(2, 1) = -2;
	kernelY.at<float>(2, 2) = -1;

	Mat kernelX;
	kernelY.copyTo(kernelX);
	transpose(kernelX, kernelX);

	filter2D(image, dX, CV_32F, kernelX);
	filter2D(image, dY, CV_32F, kernelX);

}

void gradientSobel(Mat& image, Mat& dX, Mat& dY, Mat& magnitude)
{
	//http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=sobel#sobel
	//calculate the gradients with OpenCV Sobel() function
	Sobel(image, dX, CV_32F, 1, 0, 3, 1.0 / 8.0, 0, BORDER_DEFAULT);
	Sobel(image, dY, CV_32F, 0, 1, 3, 1.0 / 8.0, 0, BORDER_DEFAULT);

	//to compute gradient magnitude:
	// http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=multiply#multiply
	// http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=add#add
	// http://docs.opencv.org/modules/core/doc/operations_on_arrays.html?highlight=sqrt#sqrt

	Mat temp(dX.rows, dY.cols, CV_32F);
	multiply(dX, dX, temp);
	temp.copyTo(magnitude);
	multiply(dY, dY, temp);
	add(temp, magnitude, magnitude);
	sqrt(magnitude, magnitude);
}

void smoothImage(Mat& image, double sigma)
{
	if (sigma <= 0.0)
	{
		return;
	}
	//smooth the image 
	// This is another example of a convolution / filtering operation, this time with a 
	// Gaussian kernel. You could also use all ones to get the mean of the pixels in the image
	GaussianBlur(image, image, Size(0, 0), sigma, sigma, BORDER_DEFAULT);
}

bool convertGradientImageForDisplay(Mat& input, Mat& output)
{
	if (input.empty())
	{
		return false;
	}
	if (output.empty() || input.rows != output.rows || input.cols != output.cols)
	{
		return false;
	}

	Mat temp; //make a copy so we don't change the input image
	input.copyTo(temp);
	//convert to range 0 - 255
	double minVal, maxVal;
	minMaxLoc(temp, &minVal, &maxVal);

	//keep zero centered at 128
	if (minVal < 0)
	{
		maxVal = max(abs(minVal), abs(maxVal));
		minVal = -maxVal;
	}

	//cout<<"Min and Max values for conversion: "<<minVal<<' '<<maxVal<<endl;

	//scale image into range [0 255] and convert from float to unsigned char
	temp = (temp - minVal) / (maxVal - minVal) * 255;
	temp.convertTo(output, CV_8UC1);
	return true;
}

bool markImageForDisplay(Mat& gray, Mat& output, Mat& mask)
{
	//duplicate gray image into three channels and place one red pixel
	vector<Mat> channels;
	for (int i = 0; i<3; i++)
	{
		channels.push_back(gray);
	}
	merge(channels, output);

	//anywhere that is marked in the mask image, suppress the green and blue 
	//channels so that the region will be highlighted red
	int numchannels = mask.channels();
	for (int row = 0; row<output.rows; row++)
	{
		unsigned char* maskPtr = mask.ptr<unsigned char>(row);
		unsigned char* imgPtr = output.ptr<unsigned char>(row);
		for (int col = 0; col<output.cols; col++)
		{
			if (maskPtr[col] == 255)
			{
				imgPtr[col * 3] = 0;
				imgPtr[col * 3 + 1] = 0;
			}
		}
	}
	return true;
}

bool readImageAndConvert(const string& filename, Mat& grayImage)
{
	//read image from file and convert to gray scale
	Mat image;
	image = imread(filename);
	if (image.empty())
	{
		return false;
	}
	cvtColor(image, grayImage, COLOR_RGB2GRAY);
	//imshow("Image View2", grayImage);
	return !grayImage.empty();
}

void onClickCallback(int event, int x, int y, int q, void* data)
{
	if (event != EVENT_LBUTTONDOWN)
		return;

	//make a square, centered on the clicked location
	for (int row = y - 10; row<y + 10; row++)
	{
		if (row < 0 || row >= maskImage.rows)
		{
			continue;
		}
		for (int col = x - 10; col<x + 10; col++)
		{
			if (col < 0 || col >= maskImage.cols)
			{
				continue;
			}
			maskImage.at<unsigned char>(row, col) = 255;
		}
	}

	markImageForDisplay(image, displayImage, maskImage);
	double dx = dX.at<float>(y, x);
	double dy = dY.at<float>(y, x);
	double magnitude = sqrt(dx*dx + dy*dy);
	double direction = atan2(dy, dx)*180.0 / M_PI;
	cout << "Point: (" << x << ", " << y << "). Gradient = (" << dx << ", " << dy << ")" << endl;
}

void onTrackbar(int value, void* data)
{
	original.copyTo(image);
	smoothImage(image, value);

	//compute the gradient four ways
	//gradient_FiniteDifferences_v1(image, dX, dY);
	//gradient_FiniteDifferences_v2(image, dX, dY);
	//gradient_OpenCV_v1(image, dX, dY);
	//gradient_OpenCV_v2(image, dX, dY);

	gradientSobel(image, dX, dY, gradientMagnitude);
	Laplacian(image, laplacian, CV_32F);
	laplacian = abs(laplacian);
	markImageForDisplay(image, displayImage, maskImage);
	convertGradientImageForDisplay(dX, displayDX);
	convertGradientImageForDisplay(dY, displayDY);
	convertGradientImageForDisplay(gradientMagnitude, displayMagnitude);
	convertGradientImageForDisplay(laplacian, displayLaplacian);
}
