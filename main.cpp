#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(){
	Mat first_image = imread("test.jpg");		//Resmi okuyoruz
	if (first_image.empty())
		return -1;

	Mat gray;
	cvtColor(first_image, gray, CV_BGR2GRAY);		//Resmi gri yapiyoruz


	Mat detected_edges;
	
	blur(gray, detected_edges, Size(3, 3));		//Blurlayip daha kolay okunur yapiyoruz
	Canny(detected_edges, detected_edges, 5, 15, 3);		//Canny kullanarak kenarlari tespit ediyoruz
	imshow("test", detected_edges);
	Mat crop;

	vector<vector<Point> > contours;
	findContours(detected_edges.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);		//Koseleri buluyoruz

	vector<Point> approx;

	float max_area = 0;		//Alan için bir deðer tanimliyoruz ve bunu buldugumuz dikdortgenlerden en buyuk alani olanini tespit etmek icin kullanicaz
	int maxi = -1;		//Bu degiskenimizde en buyuk alani olaniýn index ini tutmak icin
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);

		if (fabs(contourArea(contours[i])) < 100 || !isContourConvex(approx))		//Alani çok kucuk olanlar ve disbukey olmayanlari eliyoruz
			continue;

		if (approx.size() == 4) {		//Dortgenleri tespit ediyoruz

			float contour_area = contourArea(contours[i]);		//Alani bulup degiskenimize koyuyoruz.
			
			if (contour_area > max_area) {		//Siradaki alan daha buyukse o en buyuk alanimiz oluyor ve index ini maxi ye atiyoruz
				max_area = contour_area;
				maxi = i;
				cout << max_area << endl << maxi << endl;
			}
			
		}
	}

	if (maxi < 0){
		std::cout << "no contour found" << std::endl;		//Hic kose bulamazsak programi sonlandiriyoruz
		waitKey(0);
		return 1;
	}

	Rect r = boundingRect(contours[maxi]);		//Buldugumuz en buyuk dorgenin alaninin cropluyoruz
	crop = first_image(r);
	
	CascadeClassifier object;
	object.load("haarcascade_frontalface_default.xml");		//Yuz tespit etmek icin gerek xml dosyasi

	imshow("first_image", first_image);
	imshow("crop", crop);
	imwrite("cropped.jpg", crop);

	Mat gray2;
	cvtColor(crop, gray2, CV_BGR2GRAY);		//Yeni croplu resmimizide gri yapýyoruz yuz tespiti icin

	Mat faces = crop.clone();

	vector<Rect> objectVector;
	object.detectMultiScale(gray2, objectVector, 1.1, 3, 0, Size(30, 30));
	int max_area2 = 0;
	int max_area_index = -1;		//Bu 2 degisken yine en buyugunu bulmak icin ayný sekilde kullaniliyor
	for (int i = 0; i<objectVector.size(); i++){
		Point pt1(objectVector[i].x + objectVector[i].width, objectVector[i].y + objectVector[i].height);
		Point pt2(objectVector[i].x, objectVector[i].y);
		rectangle(faces, pt1, pt2, cvScalar(0, 255, 0, 0), 2, 8, 0);		//Yuzleri dortgen icine aliyoruz
		if (objectVector[i].width * objectVector[i].height > max_area2)
			max_area_index = i;
	}

	if (max_area_index != -1) {
		imshow("Faces", faces);		//Yuzlerin oldugu resmi cikartiyoruz

		Rect face_crop;		//En buyuk yuzu kenarlarýndan ekstra boyutla birlikte kesiyoruz
		face_crop.x = objectVector[max_area_index].x - 10;
		face_crop.y = objectVector[max_area_index].y - 30;
		face_crop.width = objectVector[max_area_index].width + 20;
		face_crop.height = objectVector[max_area_index].height + 60;

		Mat face = crop(face_crop);
		imshow("Face", face);		//En son yuzu cikartiyoruz
	}
	
	string cropped = "test_yazi.jpg";

	waitKey(0);
	return 0;
}
