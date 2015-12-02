
#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <vector>
#include <numeric>
#include <omp.h>  
#include "plot.h"
#include <fstream>

#include "ctime" 

using namespace cv;
using namespace std;

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

//Descriptor Operation
void descriptor_write(Mat descriptors, string output_file);
Mat descriptor_read(string input_file);

//Preprosessing
void preprocessing(char* video_path, char* output_folder_name);
void PersistOnDisk(char* movie_name, string file_name);
void PersistOnDisk(VideoCapture capture, string file_name);

//Query - Step 1 Compare samples with history 
int CompareMatWithHistory(Mat query_frame, string file_name, int start_frame, int end_frame, int query_sample_number, int& max_similar_rate);

//Query - Step 2 Alignment
int AlignmentFrameWithinMovie(Mat query_mat, char* movie_path, int raw_frame_index, int downsample_rate);

//Query - Step 3 Binary Search Border
int BinarySearchBorderLeft(VideoCapture query_cap, VideoCapture reference_cap,int aligned_query_begin_frame, int total_query_frame, int raw_reference_frame, int query_fps, int reference_fps, char* movie_folder_name);
int BinarySearchBorderRight(VideoCapture query_cap, VideoCapture reference_cap,int aligned_query_begin_frame, int total_query_frame, int raw_reference_frame, int query_fps, int reference_fps, char* movie_folder_name);

//Show Result Concat two Relevant Movie and BroadCast
void PlayTwoMovie(VideoCapture query, VideoCapture reference, int start_query, int start_reference, int block_size);




vector< DMatch > CompareDescriptorWithDiscriptor(Mat query_descriptor, Mat reference_descriptor, vector<KeyPoint> query_keypoint, vector<KeyPoint> reference_keypoint);
Rect get_croped_mat(Mat input_mat);

template <class T> 
  std::string ConvertToString(T value) {
  std::stringstream ss;
  ss << value;
 return ss.str();
}

template <typename T>
cv::Mat plotGraph(std::vector<T>& vals, int YRange[2])
{

    auto it = minmax_element(vals.begin(), vals.end());
    float scale = 1./ceil(*it.second - *it.first); 
    float bias = *it.first;
    int rows = YRange[1] - YRange[0] + 1;
    cv::Mat image = Mat::zeros( rows, vals.size(), CV_8UC3 );
    image.setTo(0);
    for (int i = 0; i < (int)vals.size()-1; i++)
    {
        cv::line(image, cv::Point(i, rows - 1 - (vals[i] - bias)*scale*YRange[1]), cv::Point(i+1, rows - 1 - (vals[i+1] - bias)*scale*YRange[1]), Scalar(255, 0, 0), 1);
    }

    return image;
}