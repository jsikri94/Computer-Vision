#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include "dirent.h"

using namespace std;
using namespace cv;


/*
The cascade classifiers that come with opencv are kept in the
following folder: bulid/etc/haarscascades
Set OPENCV_ROOT to the location of opencv in your system
*/
string OPENCV_ROOT = "C:/Users/Jaahanavee Sikri/Downloads/opencv/";
string cascades = OPENCV_ROOT + "build/etc/haarcascades/";
string FACES_CASCADE_NAME = cascades + "haarcascade_frontalface_alt2.xml";

/*  The mouth cascade is assumed to be in the local folder */ //what local folder?
string MOUTH_CASCADE_NAME = cascades + "Mouth.xml";
//string MOUTH_CASCADE_NAME = cascades + "haarcascade_smile.xml";


void drawRectangle(Mat frame, const Rect rect, int r, int g, int b) {
	//int width2 = rect.width / 2;
	//int height2 = rect.height / 2;
	//Point center(rect.x + width2, rect.y + height2);
	rectangle(frame, rect,
		Scalar(r, g, b), 2, 8, 0);
}


bool detectSilence(Mat frame, Point location, Mat ROI, CascadeClassifier cascade)
{
	// frame,location are used only for drawing the detected mouths
	vector<Rect> mouths;
	cascade.detectMultiScale(ROI, mouths, 1.05, 3, 0, Size(23, 30));//(1.05,3,0,23,23) better but still detecting false mouths.

	int nmouths = (int)mouths.size();
	for (int i = 0; i < nmouths; i++) {
		Rect mouth_i = mouths[i];
		drawRectangle(frame, mouth_i + location, 255, 255, 0);
	}
	return(nmouths == 0);
}

// you need to rewrite this function
int detect(Mat frame,
	CascadeClassifier cascade_face, CascadeClassifier cascade_mouth) {
	Mat frame_gray;
	vector<Rect> faces;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);

	//equalizeHist(frame_gray, frame_gray); // input, outuput
	//medianBlur(frame_gray, frame_gray, 3); // input, output, neighborhood_size
	//  blur(frame_gray, frame_gray, Size(5,5), Point(-1,-1));
	/*  input,output,neighborood_size,center_location (neg means - true center) */


	cascade_face.detectMultiScale(frame_gray, faces,
		1.05, 3, 0 | 0, Size(55, 55));//size(30,30) detects all faces with 1.1,3

									  /* frame_gray - the input image
									  faces - the output detections.
									  1.1 - scale factor for increasing/decreasing image or pattern resolution
									  3 - minNeighbors.
									  larger (4) would be more selective in determining detection
									  smaller (2,1) less selective in determining detection
									  0 - return all detections.
									  0|CV_HAAR_SCALE_IMAGE - flags. This flag means scale image to match pattern
									  Size(30, 30)) - size in pixels of smallest allowed detection
									  */

	int detected = 0;

	int nfaces = (int)faces.size();
	for (int i = 0; i < nfaces; i++) {
		Rect face = faces[i];
		drawRectangle(frame, face, 255, 0, 255);
		Rect lowerface;
		lowerface.width = 3 * face.width / 5;
		lowerface.height = face.height / 3;
		lowerface.x = face.x + face.width / 5;
		lowerface.y = face.y + 2 * face.height / 3;
		drawRectangle(frame, lowerface, 0, 0, 0);
		Mat lower_faceROI = frame_gray(lowerface);
		//equalizeHist(lower_faceROI, lower_faceROI);
		//medianBlur(lower_faceROI, lower_faceROI, 5);
		//blur(lower_faceROI, lower_faceROI, Size(3, 3), Point(-1, -1));
		if (detectSilence(frame, Point(lowerface.x, lowerface.y), lower_faceROI, cascade_mouth)) {
			drawRectangle(frame, face, 0, 255, 0);
			detected++;
		}
	}
	return(detected);
}

int runonFolder(const CascadeClassifier cascade1,
	const CascadeClassifier cascade2,
	string folder) {
	if (folder.at(folder.length() - 1) != '/') folder += '/';
	DIR *dir = opendir(folder.c_str());
	if (dir == NULL) {
		cerr << "Can't open folder " << folder << endl;
		exit(1);
	}
	bool finish = false;
	string windowName;
	struct dirent *entry;
	int detections = 0;
	while (!finish && (entry = readdir(dir)) != NULL) {
		char *name = entry->d_name;
		string dname = folder + name;
		Mat img = imread(dname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
		if (!img.empty()) {
			int d = detect(img, cascade1, cascade2);
			cerr << d << " detections" << endl;
			detections += d;
			if (!windowName.empty()) destroyWindow(windowName);
			windowName = name;
			namedWindow(windowName.c_str(), CV_WINDOW_AUTOSIZE);
			imshow(windowName.c_str(), img);
			int key = cvWaitKey(0); // Wait for a keystroke
			switch (key) {
			case 27: // <Esc>
				finish = true; break;
			default:
				break;
			}
		} // if image is available
	}
	closedir(dir);
	return(detections);
}

void runonVideo(const CascadeClassifier cascade1,
	const CascadeClassifier cascade2) {
	VideoCapture videocapture(0);
	if (!videocapture.isOpened()) {
		cerr << "Can't open default video camera" << endl;
		exit(1);
	}
	string windowName = "Live Video";
	namedWindow("video", CV_WINDOW_AUTOSIZE);
	Mat frame;
	bool finish = false;
	while (!finish) {
		if (!videocapture.read(frame)) {
			cout << "Can't capture frame" << endl;
			break;
		}
		detect(frame, cascade1, cascade2);
		imshow("video", frame);
		if (cvWaitKey(30) >= 0) finish = true;
	}
}

int main(int argc, char** argv) {
	if (argc != 1 && argc != 2) {
		cerr << argv[0] << ": "
			<< "got " << argc - 1
			<< " arguments. Expecting 0 or 1 : [image-folder]"
			<< endl;
		return(-1);
	}

	string foldername = (argc == 1) ? "" : argv[1];
	CascadeClassifier faces_cascade, mouth_cascade;

	if (
		!faces_cascade.load(FACES_CASCADE_NAME)
		|| !mouth_cascade.load(MOUTH_CASCADE_NAME)) {
		cerr << FACES_CASCADE_NAME << " or " << MOUTH_CASCADE_NAME
			<< " are not in a proper cascade format" << endl;
		return(-1);
	}

	int detections = 0;
	if (argc == 2) {
		detections = runonFolder(faces_cascade, mouth_cascade, foldername);
		cout << "Total of " << detections << " detections" << endl;
	}
	else runonVideo(faces_cascade, mouth_cascade);

	return(0);
}