#include "com_cabatuan_lbpfeatures_MainActivity.h"
#include <android/log.h>
#include <android/bitmap.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits>

#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#define  LOG_TAG    "LBPFaceDetection"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  DEBUG 0

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

Mat getLBP(Mat img){

    // Pad input with zero so output is same size
    Mat padded(img.rows+2, img.cols+2, CV_8UC1);
    copyMakeBorder(img, padded, 1, 1, 1, 1, BORDER_REPLICATE);

    Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC1);

    for(int i=1;i<padded.rows-1;i++) {
        for(int j=1;j<padded.cols-1;j++) {
            uchar center = padded.at<uchar>(i,j);
            unsigned char code = 0;

            code |= ((padded.at<uchar>(i-1,j-1)) > center) << 7;
            code |= ((padded.at<uchar>(i-1,j))   > center) << 6;
            code |= ((padded.at<uchar>(i-1,j+1)) > center) << 5;
            code |= ((padded.at<uchar>(i,j+1))   > center) << 4;
            code |= ((padded.at<uchar>(i+1,j+1)) > center) << 3;
            code |= ((padded.at<uchar>(i+1,j))   > center) << 2;
            code |= ((padded.at<uchar>(i+1,j-1)) > center) << 1;
            code |= ((padded.at<uchar>(i,j-1))   > center) << 0;
            dst.at<uchar>(i-1,j-1) = code;
        }
    }
    return dst;
}



/// 16.7 fps ave. after 500 frames using Mat pointer
//Mat *pLBP = NULL;


/// 17.2 fps ave. after 500 frames using just Mat
Mat LBP;

// FAST: almost 20 fps ave. after 500 frames
vector<Mat> channels;

// Slower at 16.1 fps?
// Mat *channels = NULL;

/*
 * Class:     com_cabatuan_lbpfeatures_MainActivity
 * Method:    predict
 * Signature: (Landroid/graphics/Bitmap;[B)V
 */
JNIEXPORT void JNICALL Java_com_cabatuan_lbpfeatures_MainActivity_predict
  (JNIEnv *pEnv, jobject clazz, jobject pTarget, jbyteArray pSource){

   AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent; // Links to Bitmap content

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   /// cv::Mat for YUV420sp source and output BGRA 
    Mat srcGray(bitmapInfo.height, bitmapInfo.width, CV_8UC1, (unsigned char *)source);
    Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);

/***********************************************************************************************/
    /// Native Image Processing HERE... 

    if (LBP.empty())
        LBP.create(srcGray.size(), srcGray.type());
    
    LBP = getLBP(srcGray); 
    
    Mat White(srcGray.size(), srcGray.type(), Scalar(255));
    
    /*
    channels.push_back(LBP);  // B
    channels.push_back(LBP);    // G
    channels.push_back(LBP);  // R
    channels.push_back(White); 
    */ 
      
    /*  NICE PINK */
    channels.push_back(White);  // B
    channels.push_back(LBP);    // G
    channels.push_back(White);  // R
    channels.push_back(White);  
                                // A 
    
                                       // transparent = 0, opaque = 255                             
    
    merge(channels, mbgra); // FAST: almost 20 fps ave. after 500 frames
    
   /* 
    if (channels == NULL)
        channels = new Mat[4];
        
    //Mat A(srcGray.size(), srcGray.type(), Scalar(255));
    
    channels[0] = LBP; 										   // R
	channels[1] = Mat::zeros(srcGray.size(), srcGray.type());  // G
    channels[2] = Mat::zeros(srcGray.size(), srcGray.type());  // B
	channels[3] = LBP;
	
	merge(channels, 4, mbgra);
    */
    
 /// Display to Android
 /// cvtColor(LBP, mbgra, CV_GRAY2BGRA);

 
/************************************************************************************************/ 
   
   /// Clear the vector (or it will accumulate!!!)
   channels.clear();
   
   /// Release Java byte buffer and unlock backing bitmap
   pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();

}
