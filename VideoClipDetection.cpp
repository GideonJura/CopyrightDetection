// VideoClipDetection.cpp : Defines the entry point for the console application.
//
#include "VideoClipDetection.h"

//Similar threshold between two different images
const double SIMILARITY_THRESHOLD = 0.10;

//Threshold of matches number between query image and refernece image
const double MATCHES_DESCRIPTOR_MIN_LIMIT = 20;

//Global Downsample rate 10 = 0.5 second
const int DOWNSAMPLE_RATE = 10; //desperated
const int DOWNSAMPLE_FRAME_PER_SECOND = 3;

//Interest Point distant threshold Range (0 -100), the smaller the similar
const int HAMMING_DISTANCE_THRESHOLD = 35; 

//Displacement Proportion between query and reference (x-axis 20%, y-axis 20%)
const double DISTANCE_OF_DISPLACEMENT_THRESHOLD = 0.20;

//Descriptor size in the Database
const int DESCRIPTOR_SIZE_X = 1280;
const int DESCRIPTOR_SIZE_Y = 720;

//Brief Compare resizing
const int BRIEF_COMPARE_SIZE_X = 320;
const int BRIEF_COMPARE_SIZE_Y = 240;

//Show Result resizing
const int SHOWOFF_COMPARE_SIZE_X = 480;
const int SHOWOFF_COMPARE_SIZE_Y = 270;

//Define Max File Block 
const int MAX_FILM_LENGTH = 10000;


//Persist descriptors to disk
//Input Mat descriptors : (input) decriptor
//Input string output_file : (input) file_path
//Return : void
void descriptor_write(Mat descriptors, string output_file){
  cv::FileStorage outfile(output_file, FileStorage::WRITE );
  outfile << "descriptor" << descriptors;
  outfile.release();
}


//Read a descriptor into a matrix
//Input string input_file : (input) input file_path
//Return : Mat the dexcriptor from input file
Mat descriptor_read(string input_file){
  Mat descriptors; 
  FileStorage fs(input_file, FileStorage::READ);
  fs["descriptor"] >> descriptors;
  fs.release();
  return descriptors;
}

//Crop the black border
//Input Mat input_mat : (input) matrix that need cropping
//Return : Rect the Rect that contains no black border
Rect get_croped_mat(Mat input_mat)
{
	Mat gray;
	cvtColor( input_mat, gray, CV_BGR2GRAY );
	cv::Scalar avgPixelIntensity = cv::mean( gray );
	
	//double xIndtensity = avgPixelIntensity.val[0];
	//imwrite("test_ori.jpg",gray);
	threshold(gray ,gray , 1, 255,THRESH_BINARY);
	//imwrite("test.jpg",gray);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, CV_RETR_EXTERNAL ,CHAIN_APPROX_SIMPLE);
	int index = 0;
	int max_num = 0;
	Rect ret_rect( 0, 0, 0, 0);

	for(int i = 0 ; i < contours.size() ; i++)
	{
		if(contours[i].size() > max_num)
		{
			index = i;
			max_num = contours[i].size();
		}
	}

	if(max_num != 0)
	{
		ret_rect = boundingRect(contours[index]);
	}
	else
	{
		ret_rect.width = input_mat.cols;
		ret_rect.height = input_mat.rows;
		ret_rect.x = 0;
		ret_rect.y =0;
	}
	return ret_rect;
}

/* Currently not used -- to Compare the difference of histogram between two images 
double HistogramCompare( Mat src_query , Mat src_reference)
{
	Mat hsv_reference;
    Mat hsv_query;

    /// Convert to HSV
    cvtColor( src_reference, hsv_reference, COLOR_BGR2HSV );
    cvtColor( src_query, hsv_query, COLOR_BGR2HSV );

    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };

    /// Histograms
    MatND hist_reference;
	MatND hist_query;

    /// Calculate the histograms for the HSV images
    calcHist( &hsv_reference, 1, channels, Mat(), hist_reference, 2, histSize, ranges, true, false );
    normalize( hist_reference, hist_reference, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_query, 1, channels, Mat(), hist_query, 2, histSize, ranges, true, false );
    normalize( hist_query, hist_query, 0, 1, NORM_MINMAX, -1, Mat() );
   
	double hist_difference =  1 - compareHist( hist_reference, hist_query, 0);

	//Make it clear

	//hist_difference = hist_difference > 0.85 ? 1 : 0 ;

	return hist_difference;

}


double HistogramCompare( char* query_file , char* reference_file)
{
    /// Load three images with different environment settings
    Mat src_reference = imread( reference_file, 1 );
	Mat src_query = imread( query_file, 1);
	return HistogramCompare(src_query, src_reference);
}
*/


double CalcMatchesProportion(vector< DMatch > good_matches, vector<KeyPoint> referenceKP, vector<KeyPoint> queryKP)
{
	double ret_value = 0.0f;

	//TODO hard_code need rewrite
	const double query_x = 856;
	const double query_y = 480;
	
	const double reference_x = DESCRIPTOR_SIZE_X;
	const double reference_y = DESCRIPTOR_SIZE_Y;
	

	int nCount = 0;
	int num_good_match = good_matches.size();

	for(int i = 0 ; i < good_matches.size(); i++)
	{
		double reference_x_prop = referenceKP[good_matches[i].trainIdx].pt.x / reference_x;
	    double reference_y_prop =  referenceKP[good_matches[i].trainIdx].pt.y / reference_y;
		double query_x_prop = queryKP[good_matches[i].queryIdx].pt.x / query_x;
		double query_y_prop = queryKP[good_matches[i].queryIdx].pt.y / query_y;

		if(abs(reference_x_prop - query_x_prop) < DISTANCE_OF_DISPLACEMENT_THRESHOLD && abs(reference_y_prop - query_y_prop))
		{
			nCount ++;
		}
	}

	if(num_good_match == 0)
	{
		ret_value = 0;
	}
	else
	{
		//ret_value = (nCount + 0.0f) / num_good_match;
		ret_value = (nCount + 0.0f) / referenceKP.size();
	}

	return ret_value;
}

vector< DMatch > CompareDescriptorWithDiscriptor(Mat query_descriptor, Mat reference_descriptor, vector<KeyPoint> query_keypoint, vector<KeyPoint> reference_keypoint)
{
	vector< DMatch > good_matches;	
	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> one_match;
	double good_match_proportion = 0;

	//TODO efficiency problem
	if(query_descriptor.cols == reference_descriptor.cols)
	{
		//matcher.match(left,right,one_match);
		matcher.match(query_descriptor,reference_descriptor,one_match);
	}
	else
	{
		return good_matches ;
	}


	for( int i = 0; i < query_descriptor.rows; i++ )
	{ 
		if( one_match[i].distance <= HAMMING_DISTANCE_THRESHOLD) 
		{ 
			good_matches.push_back(one_match[i]); 
		}
	}

	/*
	if(query_keypoint.size() == 0 || reference_keypoint.size() == 0)
	{
		good_match_proportion = 0;
		cout<<"No Keypoint @"<<endl;
	}
	else
	{
		cout<<CalcMatchesProportion(good_matches,reference_keypoint,query_keypoint)<<endl;
	}
	*/
	return good_matches;
}

//query_img input query image matrix(INPUT)
//reference_img reference image matrix(INPUT)
double CompareBrief(Mat query_img, Mat reference_img)
{
	Mat transfered_query_img;
	if(query_img.cols == 0 || query_img.rows == 0)
	{
		return -1;
	}
	if(reference_img.cols == 0 || reference_img.rows == 0)
	{
		return -1;
	}

	//Mat transfered_query_img,query_img,reference_img;
	resize(query_img,query_img,cvSize(BRIEF_COMPARE_SIZE_X,BRIEF_COMPARE_SIZE_Y),0,0,INTER_LINEAR);
	resize(reference_img, reference_img,cvSize(BRIEF_COMPARE_SIZE_X,BRIEF_COMPARE_SIZE_Y),0,0,INTER_LINEAR);

	//TODO I really need Resize? because you have compare the keypoint? Still questioned
	/*
	if(query_img.cols != reference_img.cols || query_img.rows != reference_img.rows)
	{
		resize(query_img,transfered_query_img,cvSize(reference_img.cols, reference_img.rows),0,0,INTER_LINEAR);
	}
	else
	*/
	{
		transfered_query_img = query_img.clone();
	}

	// -- Step 1: Detect the keypoints using STAR Detector 
    vector<KeyPoint> query_keypoint; 
	OrbFeatureDetector detector;
    detector.detect(transfered_query_img, query_keypoint);

	BriefDescriptorExtractor brief; 
    Mat query_descriptor; 
    brief.compute(transfered_query_img, query_keypoint, query_descriptor);

	vector<KeyPoint> reference_keypoint; 
	detector.detect(reference_img, reference_keypoint); 
	Mat reference_descriptor; 
	brief.compute(reference_img, reference_keypoint, reference_descriptor); 


	BFMatcher matcher(NORM_HAMMING);
	vector<DMatch> one_match;
	vector< DMatch > good_matches;	
	good_matches = CompareDescriptorWithDiscriptor(query_descriptor,reference_descriptor,query_keypoint,reference_keypoint);

	//Two metrics good_match_proportion: proportion of good match that distance is shorter than
	//double good_match_proportion = CalcMatchesProportion(good_matches,reference_keypoint,query_keypoint);
	//double good_match_proportion =  good_matches.size() / reference_descriptor.
	
	int ret_matches = good_matches.size();
	//DEBUG PRINT
	//cout<<ret_matches<<" "<<query_descriptor.rows<<" "<<reference_descriptor.rows<<endl;
	double good_match_proportion = (ret_matches + 0.0) / (reference_descriptor.rows + 1.0);
	
	/*
	// -- show , could delete if go to practice	
	Mat img_matches; 
	drawMatches(query_img, query_keypoint, reference_img , reference_keypoint, good_matches, img_matches); 	
	char str_i[4];
	_itoa_s(ret_matches ,str_i,10);
	putText(img_matches, str_i, cvPoint( 320,20 ),CV_FONT_HERSHEY_COMPLEX,0.7,Scalar(0,255,0));
	_itoa_s(query_keypoint.size(),str_i,10);
	putText(img_matches, str_i, cvPoint( 320,40 ),CV_FONT_HERSHEY_COMPLEX,0.7,Scalar(0,255,0));
	cvWaitKey(50);
	imshow("Result", img_matches);
	*/

	return good_match_proportion;

}

void PersistOnDisk(char* movie_name, string file_name, int downsample_rate)
{
	cv::VideoCapture capture(movie_name);  

    if (!capture.isOpened()) {  
        cout<<"ERROR"<<endl; 
    }  

	PersistOnDisk(capture, file_name, downsample_rate);
}

void PersistOnDisk(VideoCapture capture, string file_name, int downsample_rate)
{
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();
 
    if (!capture.isOpened()) {  
        cout<<"ERROR"<<endl; 
		return;
    }  

	long cur_index = 0;
	Mat store_mat;
	long total_frame=static_cast<long>(capture.get(CV_CAP_PROP_FRAME_COUNT));
	while (cur_index < total_frame) {

		if (!capture.read(store_mat)) {
			//Error
			cout<<"read Store_mat failed"<<endl;
			return;
		}

		if(cur_index % downsample_rate != 0)
		{
			cur_index ++;
			continue;
		}

		Rect rect = get_croped_mat(store_mat);
		Mat croped_mat = store_mat(rect);

		if(croped_mat.cols == 0 || croped_mat.rows == 0)
		{
			cout<<"Error @ PersistOnDisk"<<endl;
			return;
		}

		resize(croped_mat,croped_mat,cvSize(BRIEF_COMPARE_SIZE_X,BRIEF_COMPARE_SIZE_Y),0,0,INTER_LINEAR);
		vector<KeyPoint> store_keypoint; 
		OrbFeatureDetector detector;
		detector.detect(croped_mat, store_keypoint);
		//detector.detect(store_mat, store_keypoint);
		BriefDescriptorExtractor brief; 
		Mat store_descriptor; 
		brief.compute(croped_mat, store_keypoint, store_descriptor);
		//brief.compute(store_mat, store_keypoint, store_descriptor);

		stringstream ss;
		ss<<cur_index;	
		string absolute_file_name = "F://" + file_name + "//" + ss.str();
		string absolute_keypoint_file_name = "F://" + file_name + "//" + ss.str() + "_KeyPoint";
		descriptor_write(store_descriptor,absolute_file_name);	
		FileStorage fs(absolute_keypoint_file_name,FileStorage::WRITE);
		write(fs, "keypoint", store_keypoint);
		fs.release();
		
		cur_index ++;
	}

	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "%f seconds\n", duration ); 
}

vector<KeyPoint> get_key_point(string file_name, int cur_index)
{
	vector<KeyPoint> mykpts2;
	stringstream ss;
	ss<<cur_index;	
	string absolute_keypoint_file_name = "F://" + file_name + "//" + ss.str() + "_KeyPoint";
	FileStorage fs2(absolute_keypoint_file_name, FileStorage::READ);
	FileNode kptFileNode = fs2["keypoint"];
	read( kptFileNode, mykpts2 );
	fs2.release();

	return mykpts2;
}

//Most Time-Consuming-Part
int CompareMatWithHistory(Mat query_frame, string file_name, int start_frame, int end_frame, int downsample_rate, int& max_similar_rate)
{
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();

	// -- Step 1: Detect the keypoints using STAR Detector 
    vector<KeyPoint> query_keypoint; 
	OrbFeatureDetector detector;
    detector.detect(query_frame, query_keypoint);
	BriefDescriptorExtractor brief; 
    Mat query_descriptor; 
    brief.compute(query_frame, query_keypoint, query_descriptor);
	Mat reference_descriptor;
	int best_matched_img = 0;
	int max_match_number = 0;

	for(int cur_index = start_frame ; cur_index <= end_frame; cur_index += downsample_rate)
	{
		stringstream ss;
		ss<<cur_index;	
		string absolute_file_name = "F://" + file_name + "//" + ss.str();
		string absolute_keypoint_file_name = "F://" + file_name + "//" + ss.str() + "_KeyPoint";

		vector<KeyPoint> reference_keypoint;
		reference_keypoint = get_key_point(file_name, cur_index);
	
		vector<KeyPoint> query_keypoint; 
		OrbFeatureDetector detector;
		detector.detect(query_frame, query_keypoint);

		reference_descriptor = descriptor_read(absolute_file_name);
		int similar_size = CompareDescriptorWithDiscriptor(query_descriptor,reference_descriptor,query_keypoint,reference_keypoint).size();
		//DEBUG PRINT
		//cout<<cur_index<<" "<<similar_size<<endl;
		if(similar_size > max_match_number)
		{
			max_match_number =  similar_size;
			best_matched_img = cur_index;
		}
	}

	max_similar_rate = max_match_number;

	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "CompareMatWithHistory : %f seconds\n", duration ); 

	return best_matched_img;
}

/*
int CompareMatWithMovie(VideoCapture capture, Mat query)
{
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();

	// -- Step 1: Detect the keypoints using STAR Detector 
    vector<KeyPoint> query_keypoint; 
	OrbFeatureDetector detector;
    detector.detect(query, query_keypoint);
	BriefDescriptorExtractor brief; 
    Mat query_descriptor; 
    brief.compute(query, query_keypoint, query_descriptor);
  
    if (!capture.isOpened()) {  
        cout<<"ERROR"<<endl; 
    }  

	const int c_nMax = 10000;
	int cur_index = 0;
	double max_match_proportion = 0;
	const int down_sample_rate = 20;
	//vector<Mat> frames;
	double X[c_nMax + 1];
	double Y[c_nMax + 1];

	Mat reference;
	Mat best_matched_img;
	Mat last_frame;
	int counter_for_plot = 0;
	double difference = 0;
	double similar = 0;
	while (cur_index < c_nMax) {
		if (!capture.read(reference)) {
			//Error
		}

		if(cur_index % down_sample_rate != 0)
		{
			cur_index ++;
			continue;
		}
		similar = CompareBrief(query, reference);
		if(similar > max_match_proportion)
		{
			max_match_proportion =  similar;
			best_matched_img = reference.clone();
		}
		cur_index++;

		//Compare

		if(cur_index < down_sample_rate - 1)
		{
			last_frame=reference.clone();
			continue;
		}
		else
		{
			difference = HistogramCompare(last_frame,reference);
			X[counter_for_plot] = cur_index;
			Y[counter_for_plot] = difference;
			counter_for_plot ++;
		}
	}

    Plot plot;
    CvScalar color = CV_RGB(255, 0, 0);
    plot.plot(X, Y, counter_for_plot - 1, color);
    cvShowImage("Difference", plot.Figure);
	
	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "%f seconds\n", duration ); 

	return similar;
}
*/
int CompareMovieWithMovie(char* reference_movie, char* query_movie, char* movie_folder_name)
{
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();
	//TODO to use a var to set the movie name!!URGENT!!
	//char* movie_folder_name = "Reference2";

	cv::VideoCapture query_cap(query_movie);  
	cv::VideoCapture reference_cap(reference_movie);

    if (!query_cap.isOpened()) {  
        cout<<"ERROR"<<endl;
		return -100;
    }

	if (!reference_cap.isOpened()) {  
        cout<<"ERROR"<<endl;
		return -100;
    }

	//We assume at least 1/10 of the Query Clip should be part of the referenceMovie
	//Get the Middle of the Query Movie

	long query_total_frame=static_cast<long>(query_cap.get(CV_CAP_PROP_FRAME_COUNT));
	long reference_total_frame = static_cast<long>(reference_cap.get(CV_CAP_PROP_FRAME_COUNT));
	Mat query_frame;
	
	long query_block_left_border[DOWNSAMPLE_RATE];
	long query_block_right_border[DOWNSAMPLE_RATE];
	bool is_frame_in_this_clip[DOWNSAMPLE_RATE];
	long best_frame_index[DOWNSAMPLE_RATE];
	long reference_start_index[DOWNSAMPLE_RATE];
	long query_start_index[DOWNSAMPLE_RATE];


	int raw_frame_index = 0;
	int good_match_number = 0;
	int longest_block_index = -1;
	int longest_block_length = 0;
	//const int head_tail_frame_count = query_total_frame /  DOWNSAMPLE_RATE ;

	//for(int i = 1 ; i < 2 ; i++)	
	for(int i = 0 ; i < DOWNSAMPLE_RATE ; i++)
	{		
		cout<<"GO TO NUM: "<<i<<" Query Frames(0 - 9)"<<endl;
		//choose the middle frame for example
		long query_begin_frame = query_total_frame / DOWNSAMPLE_RATE * i  + i;
		query_cap.set( CV_CAP_PROP_POS_FRAMES,query_begin_frame); 
		query_cap.read(query_frame);
		//Remove Black Border
		Rect rect = get_croped_mat(query_frame);
		Mat croped_frame = query_frame(rect);
		if(croped_frame.cols == 0 || croped_frame.rows == 0)
		{
			cout<<"Error @ CompareMovieWithMovie"<<endl;
			continue;
		}

		resize(croped_frame,croped_frame,cvSize(BRIEF_COMPARE_SIZE_X,BRIEF_COMPARE_SIZE_Y), 0, 0,INTER_LINEAR);
		imwrite("test_croped.jpg",croped_frame);	
		raw_frame_index = CompareMatWithHistory(croped_frame, movie_folder_name, 0 , reference_total_frame, DOWNSAMPLE_RATE,good_match_number);

		if(good_match_number < MATCHES_DESCRIPTOR_MIN_LIMIT)
		{
			cout<<"NOT FOUND"<<endl;
			is_frame_in_this_clip[i] = false;
			continue;
		}
		
		is_frame_in_this_clip[i] = true;

		int off_size = AlignmentFrameWithinMovie(query_frame, reference_movie, raw_frame_index, DOWNSAMPLE_RATE);
		cout<<"Alignment_Off_Size: "<<off_size<<endl;

		//query_begin_frame 1464 total frame 2928 raw_frame 3940 off_size 1 block 20
		int query_clip_begin_block = BinarySearchBorderLeft(query_cap,reference_cap, query_begin_frame - off_size, query_total_frame, raw_frame_index, DOWNSAMPLE_RATE, movie_folder_name);

		//query_begin_frame 1464 total frame 2928 raw_frame 3940 off_size = 1
		int query_clip_end_block = BinarySearchBorderRight(query_cap,reference_cap, query_begin_frame - off_size, query_total_frame, raw_frame_index, DOWNSAMPLE_RATE, movie_folder_name);

		int blocks = query_clip_end_block - query_clip_begin_block;
		query_start_index[i] = (query_begin_frame - off_size) % DOWNSAMPLE_RATE + DOWNSAMPLE_RATE * query_clip_begin_block;
		reference_start_index[i] = raw_frame_index - (query_begin_frame - off_size - query_start_index[i]);

		query_block_left_border[i] = query_clip_begin_block;
		query_block_right_border[i] = query_clip_end_block;

		if(query_clip_end_block - query_clip_begin_block > longest_block_length)
		{
			longest_block_length = query_clip_end_block - query_clip_begin_block;
			longest_block_index = i;
		}
	}

	cin.get();

	for(int j = 0 ; j < DOWNSAMPLE_RATE ; j++)
	{
		cout<<j<<" "<<query_block_left_border[j]<<" "<<query_block_right_border[j]<<" GAP:"<<query_block_right_border[j] - query_block_left_border[j]<<endl;
		cout<<"Reference Start"<<reference_start_index[j]<<" Query Start "<< query_start_index[j]<<endl;
	}

	//Find a block that conatins reference movie
	if(longest_block_index  >= 0)
	{
		//Find the block from the 10 "Keyframes"
		//Step 1 Use the longest block
		int reference_start_max_frame = reference_start_index[longest_block_index];
		int query_start_max_frame = query_start_index[longest_block_index];

		int query_begin_max_block = query_block_left_border[longest_block_index];
		int query_end_max_block = query_block_right_border[longest_block_index];

		const int merge_clip_max_offset_threshold = 2;
		const int very_short_clip_block_threshold = 1;
		//Step 2 extend left or right border
		for(int k = 0 ; k < DOWNSAMPLE_RATE ; k++)
		{
			//if the block is very short, ignore this block
			if(query_block_right_border[k] - query_block_right_border[k] <= very_short_clip_block_threshold) 
			{
				continue;
			}
			//left border should be extend
			if(query_block_left_border[k] < query_begin_max_block)
			{
				//Validate
				if( abs((query_block_left_border[k] - query_begin_max_block) - (query_start_index[k] - query_start_max_frame) / DOWNSAMPLE_RATE) < merge_clip_max_offset_threshold)
				{
					reference_start_max_frame = reference_start_index[k];
					query_start_max_frame = query_start_index[k];
					query_begin_max_block = query_block_left_border[k];
				}
			}

			//right border should be extend
			if(query_block_right_border[k] > query_end_max_block)
			{
				//Validate
				if( abs((query_block_left_border[k] - query_begin_max_block) - (query_start_index[k] - query_start_max_frame) / DOWNSAMPLE_RATE) < merge_clip_max_offset_threshold)
				{
					reference_start_max_frame = reference_start_index[k];
					query_start_max_frame = query_start_index[k];
					query_end_max_block = query_block_right_border[k];
				}
			}
		}

		int max_block_extended = query_end_max_block -	query_begin_max_block;
		const int frame_per_second = 30;

		cout<<"#######################################"<<endl;
		cout<<"#FINAL RESULT"<<"Query Frame Come From:"<<query_start_max_frame<<endl;
		cout<<"#FINAL RESULT"<<":Query_Block_Begin:"<<query_begin_max_block<<endl;
		cout<<"#FINAL RESULT"<<":Query_Block_End:"<<query_end_max_block<<endl;
		cout<<"#FINAL RESULT"<<":Total Time:"<<(query_end_max_block - query_begin_max_block) * DOWNSAMPLE_RATE  / frame_per_second<<"SECOND"<<endl;
		cout<<"#######################################"<<endl;
		PlayTwoMovie(query_cap,reference_cap,query_start_max_frame,reference_start_max_frame,max_block_extended,DOWNSAMPLE_RATE);
	}
	else
	{
		//Not Find
		cout<<"#######################################"<<endl;
		cout<<"#				NOT FOUND			 #"<<endl;
		cout<<"#######################################"<<endl;
		
	}
	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "CompareMovieWithMovie : %f seconds\n", duration ); 

	cin.get();

	return 0;
}

void PlayTwoMovie(VideoCapture query_cap, VideoCapture reference_cap, int start_query, int start_reference, int blocks,int downsample_rate)
{
	Mat query_mat, reference_mat, concat_mat;
	for(int i = 0 ; i < blocks ; i++)
	{
		if (i == 0)
		{
			imwrite("query_start.jpg",query_mat);
			imwrite("reference_start.jpg",reference_mat);
		}

		if (i == blocks - 1)
		{
			imwrite("query_end.jpg",query_mat);
			imwrite("reference_end.jpg",reference_mat);
		}

		int frame_in_the_reference = start_reference + i * downsample_rate;
		int frame_in_the_query = start_query + i * downsample_rate;
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query);
		query_cap.read(query_mat);
		reference_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_reference);
		reference_cap.read(reference_mat);
		if(query_mat.cols == 0 || query_mat.rows == 0)
		{
			continue;
		}
		if(reference_mat.cols == 0 || reference_mat.rows == 0)
		{
			continue;
		}

		//Go to 16:9
		resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y),0,0,1);
		resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y),0,0,1);
		//Vertically Concat two Mat
		hconcat(query_mat,reference_mat,concat_mat);
		imshow("Compare",concat_mat);
		cvWaitKey(50);
	}
}

//2828, 5656, 7880, -5, xxx.mp4
//2828 - 5 = 2323
//2323 = 20 * 116 block
int BinarySearchBorderLeft(VideoCapture query_cap, VideoCapture reference_cap,int aligned_query_begin_frame, int total_query_frame, int raw_reference_frame, int downsample_rate, char* movie_folder_name)
{
	int total_number_of_block = aligned_query_begin_frame / downsample_rate;
	int reference_start = raw_reference_frame - total_number_of_block * downsample_rate;
	int alignment_query_start = aligned_query_begin_frame - total_number_of_block * downsample_rate;
	//find the first block match the refernece
	int start_block = 0;
	int end_block = aligned_query_begin_frame / downsample_rate;
	//Think of the beginning of the movie may be the advertisement, or intro, we use step to get close to the front
	double difference = 0;
	Mat query_mat,reference_mat,concat_mat;
	while(end_block > start_block)
	{
		//Try Half way of the movie
		int cur_block = ceil((start_block + end_block + 0.0) / 2);
		int frame_in_the_reference = reference_start + cur_block * downsample_rate;
		int frame_in_the_query = alignment_query_start + cur_block * downsample_rate;

		//TODO right border
		if( frame_in_the_reference < 0 ||  frame_in_the_query < 0)
		{
			if(end_block - start_block == 1)
			{
				return end_block;
			}
			else
			{
				start_block = cur_block;
				continue;
			}
		}
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query);
		query_cap.read(query_mat);
		reference_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_reference);
		reference_cap.read(reference_mat);

		if(query_mat.cols == 0 || query_mat.rows == 0 || reference_mat.cols == 0 || reference_mat.rows == 0)
		{
			if(end_block - start_block == 1)
			{
				return end_block;
			}
			else
			{
				start_block = cur_block;
				continue;
			}
		}
		
		//Go to 16:9
		resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y),0,0,1);
		resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y),0,0,1);
		/*
		//Vertically Concat two Mat
		hconcat(query_mat,reference_mat,concat_mat);
		imshow("Left Border Search",concat_mat);
		cvWaitKey(50);
		*/

		double similar_result = CompareBrief(query_mat,reference_mat);
		if(similar_result > SIMILARITY_THRESHOLD)
		{
			if(end_block - start_block == 1)
			{
				return start_block;
			}
			else
			{
				end_block = cur_block;
			}
		}
		else
		{
			if(end_block - start_block == 1)
			{
				return end_block;
			}
			else
			{
				start_block = cur_block;
			}
		}
	}

	return end_block;
}

//2828, 5656, 7880, -5, xxx.mp4
//2828 - 5 = 2323
//2323 = 20 * 116 block
int BinarySearchBorderRight(VideoCapture query_cap, VideoCapture reference_cap,int aligned_query_begin_frame, int total_query_frame, int raw_reference_frame, int downsample_rate, char* movie_folder_name)
{
	int total_number_of_block = aligned_query_begin_frame / downsample_rate;
	int reference_start = raw_reference_frame - total_number_of_block * downsample_rate;
	int alignment_query_start = aligned_query_begin_frame - total_number_of_block * downsample_rate;


	//find the first block match the refernece
	int start_block = aligned_query_begin_frame / downsample_rate;
	int end_block = total_query_frame / downsample_rate;
	//Think of the beginning of the movie may be the advertisement, or intro, we use step to get close to the front
	Mat query_mat,reference_mat,concat_mat;

	int reference_frame_total = reference_cap.get(CV_CAP_PROP_FRAME_COUNT);
	while(end_block > start_block)
	{
		//Try Half way of the movie
		int cur_block = ceil((start_block + end_block + 0.0) / 2);
		int frame_in_the_reference = reference_start + cur_block * downsample_rate;
		int frame_in_the_query = alignment_query_start + cur_block * downsample_rate;
		/*
		if( frame_in_the_reference < 0 ||  frame_in_the_query < 0)
		{
			if(end_block - start_block == 1)
			{
				return start_block;
			}
			else
			{
				end_block = cur_block;
				continue;
			}
		}
		*/
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query);
		query_cap.read(query_mat);

		if(frame_in_the_reference > reference_frame_total)
		{
			if(end_block - start_block == 1)
			{
				return start_block;
			}
			else
			{
				end_block = cur_block;
			}
			continue;
		}

		reference_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_reference);
		reference_cap.read(reference_mat);
		
		if(query_mat.cols == 0 || query_mat.rows == 0 || reference_mat.cols == 0 || reference_mat.rows == 0)
		{
			//Not Found
			if(end_block - start_block == 1)
			{
				return start_block;
			}
			else
			{
				end_block = cur_block;
				continue;
			}
		}

		//Go to 16:9
		resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y),0,0,1);
		resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y),0,0,1);
		/*
		//Vertically Concat two Mat
		hconcat(query_mat,reference_mat,concat_mat);
		imshow("Right Border Search",concat_mat);
		cvWaitKey(50);
		*/

		double similar_result = CompareBrief(query_mat,reference_mat);
		if(similar_result > SIMILARITY_THRESHOLD)
		{
			if(end_block - start_block == 1)
			{
				return end_block;
			}
			else
			{
				start_block = cur_block;
			}
		}
		else
		{
			if(end_block - start_block == 1)
			{
				return start_block;
			}
			else
			{
				end_block = cur_block;
			}
		}
	}

	return start_block;
}

/*
void CompareMatWithMovie(char* movie_path, char* picture_path, char* title)
 {
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();

	Mat query = imread(picture_path);
	cv::VideoCapture capture(movie_path);  
  
    if (!capture.isOpened()) {  
        cout<<"ERROR"<<endl; 
    }  

	CompareMatWithMovie(capture,query);

	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "%f seconds\n", duration ); 
}


void CompareSIFT(Mat img, Mat img2)
{
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();

	if(img.empty())
	{
		return;
	}
	if(img2.empty())
	{
		return;
	}
	
	SiftFeatureDetector siftdtc;
	vector<KeyPoint> kp1, kp2;

	siftdtc.detect(img, kp1);
	Mat outimg1;
	drawKeypoints(img, kp1, outimg1);

	siftdtc.detect(img2, kp2);
	Mat outimg2;
	drawKeypoints(img2, kp2, outimg2);
	
	SiftDescriptorExtractor extractor;
	Mat descriptor1, descriptor2;

	//Step 2 : Get Descriptors
	extractor.compute(img, kp1, descriptor1);
	extractor.compute(img2, kp2, descriptor2);


	int k = 10;
	//cv::flann::Index flannIndex(descriptor1, cv::flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_EUCLIDEAN);
	cv::flann::Index flannIndex(descriptor1, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_EUCLIDEAN);
	flannIndex.save("F:\\templsh.txt");
	Mat indices, dists;
	flannIndex.radiusSearch(descriptor2,indices,dists,10,3,flann::SearchParams(-1));

	imshow("result", indices);
	imshow("dists", dists);

	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "%f seconds\n", duration ); 
}
*/

void validateFrame(Mat* frame, int validate_frame_number)
{
	;
}

void preprocessing(char* video_path, char* output_folder_name)
{
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();

	/*
	char* video_path = "F:\\referenceMKV.mp4";
	PersistOnDisk(video_path,"007",20);

	video_path = "F:\\GOT01.mp4";
	PersistOnDisk(video_path,"ThroneOfPower01",20);

	video_path = "F:\\GOT02.mp4";
	PersistOnDisk(video_path,"ThroneOfPower02",20);

	video_path = "F:\\GOT03.mp4";
	PersistOnDisk(video_path,"ThroneOfPower03",20);
	*/

	PersistOnDisk(video_path,output_folder_name,DOWNSAMPLE_RATE);

	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "preprocessing : %f seconds\n", duration ); 

}

int AlignmentFrameWithinMovie(Mat query_mat, char* movie_path, int raw_frame_index, int downsample_rate)
{
	cv::VideoCapture reference_cap(movie_path);    
  
    if (!reference_cap.isOpened()) {  
        cout<<"ERROR"<<endl; 
    }  

	long start_frame = (raw_frame_index - downsample_rate) >= 0 ? (raw_frame_index - downsample_rate) : 0;
	long end_frame   = (raw_frame_index + downsample_rate) <= 1000000 ? (raw_frame_index + downsample_rate) : CV_CAP_PROP_FRAME_COUNT;
	
	int offsize = 0;
	double best_match_proprotion = 0;
	for(long curFrame = start_frame + 1; curFrame< end_frame; curFrame ++)
	{
		Mat reference_mat;
		reference_cap.set( CV_CAP_PROP_POS_FRAMES,curFrame); 
		reference_cap.read(reference_mat);
		double x = 0;
		double good_match_proportion = CompareBrief(query_mat, reference_mat);
		if(good_match_proportion > best_match_proprotion)
		{
			best_match_proprotion = good_match_proportion;
			offsize = curFrame - raw_frame_index;
		}
		cout<<curFrame<<" "<<good_match_proportion<<endl;
	}

	return offsize;
}

void test()
{
	//char* strLeft = "F:\\test_1.jpg";
	//char* strRight = "F:\\test_2.jpg";

	//char* strLeft = "left.jpg";
	//char* strRight = "right.jpg";

	char* strLeft = "compress50.jpg";
	char* strRight = "original1.jpg";

	//char* strLeft = "test-cici1.png";
	//char* strRight = "test-cici2.png";

	Mat img1 = imread(strLeft);
	Mat img2 = imread(strRight);

	//CompareSIFT(img1, img2);
	//CompareBrief(img1, img2,0,"Briegf");

	char* video_path = "F:\GOT02.mp4";
	char* bmp_path = "GOT02[00_02_38][20151104-125336-1].BMP";
	//CompareMatWithMovie(video_path, bmp_path,video_path);

	//Test Hisogram
	//double result = HistogramCompare(strLeft,strRight);
	//cout<<result<<endl;

	char* video_path1 = "F:\GOT01.mp4";
	char* video_path2 = "F:\GOT02.mp4";

	//char* video_path1 = "F:\referenceMKV.mp4";
	//char* video_path2 = "F:\clip1.mp4";

	//CompareMovieWithMovie(video_path1,video_path2);

	Mat x = imread(bmp_path);

	/*
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			cout<<CompareMatWithHistory(x,"ThroneOfPower01");
		}
		#pragma omp section
		{
			cout<<CompareMatWithHistory(x,"ThroneOfPower02");
		}
		#pragma omp section
		{
			cout<<CompareMatWithHistory(x,"ThroneOfPower03");
		}
		#pragma omp section
		{
			cout<<CompareMatWithHistory(x,"Unicorn01");
		}
	}
	*/
	//cout<<CompareMatWithHistory(x,"ThroneOfPower01")<<endl;
	//cout<<CompareMatWithHistory(x,"ThroneOfPower03")<<endl;
	//cout<<CompareMatWithHistory(x,"Unicorn01")<<endl;
	/*
	int raw_frame = CompareMatWithHistory(x,"ThroneOfPower02", 3000 , 6000 , 20);
	cout<<"Raw_Frame: "<<raw_frame<<endl;
	int off_size = AlignmentFrameWithinMovie(x, video_path2, raw_frame, 20);
	cout<<"Alignment_Off_Size: "<<off_size<<endl;
	*/
	/*
	char* reference_clip = "F:\\referenceMKV.mp4";
	char* query_clip = "F:\\clip1.mp4";
	char* query_clip1 = "F:\\clip1_320240.mp4";
	char* query_clip2 = "F:\\clip2_320240.mp4";
	char* query_clip3 = "F:\\clip3_320240.mp4";
	CompareMovieWithMovie(reference_clip, query_clip);
	*/
}


int _tmain(int argc, _TCHAR* argv[])
{

	//test();
	/*
	char* reference_clip = "F:\\Reference2.mp4";
	char* query_clip1 = "F:\\A+B+C\\1280720\\A+B+C.mp4";
	char* query_clip2 = "F:\\A+B+C\\720576\\A+B+C_720576.mp4";
	char* query_clip3 = "F:\\A+B+C\\320240\\A+B+C_320240.mp4";
	char* query_clip4 = "F:\\A+B+C\\720480\\A+B+C_720480.mp4";
	char* query_clip5 = "F:\\A+B+C\\400240\\A+B+C_400240.mp4";
	char* query_clip6 = "F:\\A+B+C\\32\\A+B+C_32.mp4";
	char* query_clip7 = "F:\\A+B+C\\43\\A+B+C_43.mp4";
	char* query_clip8 = "F:\\A+B+C\\54\\A+B+C_54.mp4";

	char* reference_folder_name = "Reference2";
	*/
	/*
	char* reference_folder_name = "EndlessLove";
	char* reference_clip_test = "F:\\core_dataset\\core_dataset\\endless_love\\242f1fb1c242b8b01b994185c0111d1ca25ee03e.mp4";
	char* query_clip_now_ok = "F:\\core_dataset\\core_dataset\\endless_love\\5b46e9007b0add1d73a11d2f4414efe35b017acb.mp4";
	preprocessing(reference_clip_test,reference_folder_name);
	CompareMovieWithMovie(reference_clip_test, query_clip_now_ok,reference_folder_name);
	*/

	/*
	char* reference_folder_name = "BeatifulMind";
	char* reference_clip_test = "F:\\core_dataset\\core_dataset\\beautiful_mind_game_theory\\Clip2_beautiful.mp4";
	char* query_clip_now_ok =   "F:\\core_dataset\\core_dataset\\beautiful_mind_game_theory\\videoplayback.mp4";
	preprocessing(reference_clip_test,reference_folder_name);
	CompareMovieWithMovie(reference_clip_test, query_clip_now_ok,reference_folder_name);
	*/

	//char* reference_clip_test = "F:\\core_dataset\\core_dataset\\endless_love\\6834f12ea580a4e7309bd8541fb6fbe4daeeda0f.mp4";
	//char* query_clip_test_1 = "F:\\core_dataset\\core_dataset\\endless_love\\f8b09ea589093ea7fc20cb530963fe610e90fd12.mp4";
	//char* query_clip_test_2 = "F:\\core_dataset\\core_dataset\\endless_love\\5b90735d0a44043685ced4a784dfd3fc8489317a.mp4";
	
	//preprocessing(reference_clip_test,reference_folder_name);
	//CompareMovieWithMovie(reference_clip_test, query_clip_test_1,reference_folder_name);

	//Test
	//char* bmp_path = 
	//Mat x = imread(bmp_path);


	char* video_path1 = "F:\GOT01.mp4";
	char* video_path2 = "F:\GOT02.mp4";
	//preprocessing(video_path1,"GameOfThroneS05Ep01");
	//preprocessing(video_path2,"GameOfThroneS05Ep02");
	CompareMovieWithMovie(video_path1,"F:\\test.mp4","GameOfThroneS05Ep01");


	cin.get();

	return EXIT_SUCCESS;
}

