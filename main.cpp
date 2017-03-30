/*#include <opencv2/opencv.hpp>  
#include <cstdio>  
#include <cstdlib>  
#include <Windows.h>  
using namespace std;  
int main()  
{  
    // ����Haar������������  
    // haarcascade_frontalface_alt.xmlϵOpenCV�Դ��ķ����� �������һ����ϵ��ļ�·��  
    const char *pstrCascadeFileName = "C:\\opencv2.4.4\\data\\haarcascades\\haarcascade_frontalface_alt2.xml";  
    CvHaarClassifierCascade *pHaarCascade = NULL;  
    pHaarCascade = (CvHaarClassifierCascade*)cvLoad(pstrCascadeFileName);  
  
    // ����ͼ��  
    const char *pstrImageName = "101.jpg";  
    IplImage *pSrcImage = cvLoadImage(pstrImageName, CV_LOAD_IMAGE_UNCHANGED);  
      
    IplImage *pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);  
    cvCvtColor(pSrcImage, pGrayImage, CV_BGR2GRAY);  
  
    // ����ʶ������  
    if (pHaarCascade != NULL)  
    {         
        CvScalar FaceCirclecolors[] =   
        {  
            {{0, 0, 255}},  
            {{0, 128, 255}},  
            {{0, 255, 255}},  
            {{0, 255, 0}},  
            {{255, 128, 0}},  
            {{255, 255, 0}},  
            {{255, 0, 0}},  
            {{255, 0, 255}}  
        };  
  
        CvMemStorage *pcvMStorage = cvCreateMemStorage(0);  
        cvClearMemStorage(pcvMStorage);  
        // ʶ��  
        DWORD dwTimeBegin, dwTimeEnd;  
        dwTimeBegin = GetTickCount();  
        CvSeq *pcvSeqFaces = cvHaarDetectObjects(pGrayImage, pHaarCascade, pcvMStorage);  
        dwTimeEnd = GetTickCount();  
  
        printf("��������: %d   ʶ����ʱ: %d ms\n", pcvSeqFaces->total, dwTimeEnd - dwTimeBegin);  
          
        // ���  
        for(int i = 0; i <pcvSeqFaces->total; i++)  
        {  
            CvRect* r = (CvRect*)cvGetSeqElem(pcvSeqFaces, i);  
            CvPoint center;  
            int radius;  
            center.x = cvRound((r->x + r->width * 0.5));  
            center.y = cvRound((r->y + r->height * 0.5));  
            radius = cvRound((r->width + r->height) * 0.25);  
            cvCircle(pSrcImage, center, radius, FaceCirclecolors[i % 8], 2);  
        }  
        cvReleaseMemStorage(&pcvMStorage);  
    }  
      
    const char *pstrWindowsTitle = "����ʶ�� ";  
    cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);  
    cvShowImage(pstrWindowsTitle, pSrcImage);  
  
    cvWaitKey(0);  
  
    cvDestroyWindow(pstrWindowsTitle);  
    cvReleaseImage(&pSrcImage);   
    cvReleaseImage(&pGrayImage);  
    return 0;  
}*/
#include<opencv2\opencv.hpp>  
#include <iostream>  
#include <stdio.h>  
  
using namespace std;  
using namespace cv;  
  
/** Function Headers */  
void detectAndDisplay(Mat frame);  
  
/** Global variables */  
String face_cascade_name = "C:\\opencv2.4.4\\data\\haarcascades\\haarcascade_frontalface_default.xml";  
String eyes_cascade_name = "C:\\opencv2.4.4\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";  
CascadeClassifier face_cascade;   //��������������  
CascadeClassifier eyes_cascade;   //�������۷�����  
String window_name = "Capture - Face detection";  
  
/** @function main */  
int main(void)  
{   
	
    Mat frame = imread("10.jpg");  
    //VideoCapture capture;  
    //Mat frame;  
    //-- 1. Load the cascades  
    if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };  
    if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };  
    //-- 2. Read the video stream  
    //capture.open(0);  
    //if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }  
  
    //while (capture.read(frame))  
    //{  
    //  if (frame.empty())  
    //  {  
    //      printf(" --(!) No captured frame -- Break!");  
    //      break;  
    //  }  
  
        //-- 3. Apply the classifier to the frame  
        //detectAndDisplay(frame);  
    std::vector<Rect> faces;  
	Mat img_gray;  
  
	cvtColor(frame, img_gray, CV_BGR2GRAY);  
	equalizeHist(img_gray, img_gray);  
  
//-- Detect faces  
	face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, Size(50, 50));  
  
	for (size_t j = 0; j < faces.size(); j++)  
	{  
		Mat faceROI = frame(faces[j]);  
		Mat MyFace;  
		Mat MyFace2;
		if (faceROI.cols > 100)  
		{   //int i=1;
			resize(faceROI, MyFace, Size(92, 112));  
			cvtColor(MyFace,MyFace2,CV_BGR2GRAY);
			string  str = format("D:\\faces\\%d.jpg", 10);  
			imwrite(str, MyFace2); 			
			imshow("ii", MyFace2);  
		}  
		waitKey(10);  
	}  

        int c = waitKey(0);  
        if ((char)c == 27) { return 0; } // escape  
    //}  
    return 0;  
	
}  
  
/** @function detectAndDisplay */  
/*void detectAndDisplay(Mat img)  
{  
    std::vector<Rect> faces;  
	Mat img_gray;  
  
	cvtColor(img, img_gray, COLOR_BGR2GRAY);  
	equalizeHist(img_gray, img_gray);  
  
//-- Detect faces  
	face_cascade.detectMultiScale(img_gray, faces, 1.1, 3, CV_HAAR_DO_ROUGH_SEARCH, Size(50, 50));  
  
	for (size_t j = 0; j < faces.size(); j++)  
	{  
		Mat faceROI = img(faces[j]);  
		Mat MyFace;  
		Mat MyFace2;
		if (faceROI.cols > 100)  
		{   //int i=1;
			resize(faceROI, MyFace, Size(92, 112));  
			cvtColor(MyFace,MyFace2,CV_BGR2GRAY);
			string  str = format("D:\\faces\\%d.jpg", k);  
			imwrite(str, MyFace2); 			
			imshow("ii", MyFace2);  
		}  
		waitKey(10);  
	}  
}  
*/
