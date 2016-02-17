// VideoClipDetection.cpp : Defines the entry point for the console application.
//
#include "VideoClipDetection.h"

//Similar threshold between two different images
const double SIMILARITY_THRESHOLD = 0.15;

//Similar threshold between two different images in validating
const double SIMILARITY_THRESHOLD_VALIDATE = SIMILARITY_THRESHOLD * 0.67;

//Validate Threshold to Recognize
const double VALIDATE_THRESHOLD = 0.4;

//Threshold of matches number between query image and refernece image
const double MATCHES_DESCRIPTOR_MIN_LIMIT = 20;

//Threshold when we find a very long clip in the query
const double LONG_MATCHES_IN_QUERY_TERMINATE = 0.95;

//Global Downsample rate 10 = 0.5 second
const int DOWNSAMPLE_RATE = 15; 

//Interest Point distant threshold Range (0 -100), the smaller the similar
const int HAMMING_DISTANCE_THRESHOLD = 35; 

//Displacement Proportion between query and reference (x-axis 20%, y-axis 20%)
//Not Used In This Version
const double DISTANCE_OF_DISPLACEMENT_THRESHOLD = 0.20;

//Descriptor size in the Database
const int DESCRIPTOR_SIZE_X = 1280;
const int DESCRIPTOR_SIZE_Y = 720;

//Brief Compare resizing
const int BRIEF_COMPARE_SIZE_X = 320;
const int BRIEF_COMPARE_SIZE_Y = 240;

//Show Result resizing
const int SHOWOFF_COMPARE_SIZE_X = 640;
const int SHOWOFF_COMPARE_SIZE_Y = 360;

//Define Default FPS
const int FRAME_PER_SECOND = 30;

//Global Experiemnt Output Path
string annotation_input_file_name =  "";
string annotation_output_file_name = "";
ofstream fout("F:\\ExpResult\\Result.txt");

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
	Rect ret_rect( 0, 0, 0, 0);

	if(input_mat.cols == 0 || input_mat.rows == 0)
	{
		return ret_rect;
	}
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
		DMatch dm(0,0,0);
		good_matches.push_back(dm);
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
double CompareBrief(Mat query_img, Mat reference_img, bool should_store=false, string file_name = "")
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
	resize(query_img,query_img,cvSize(533,800),0,0,INTER_LINEAR);
	resize(reference_img, reference_img,cvSize(533,800),0,0,INTER_LINEAR);

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
	
	 Mat img_keypoints_1; Mat img_keypoints_2;

	 drawKeypoints( transfered_query_img, query_keypoint, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	 imwrite("Keypoints_1.jpg", img_keypoints_1 );

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
	
	
	if(should_store)
	{
		// -- show , could delete if go to practice	
		Mat img_matches; 
		if(query_keypoint.size() == 0 || reference_keypoint.size() == 0)
		{
			return good_match_proportion;
		}
		drawMatches(query_img, query_keypoint, reference_img , reference_keypoint, good_matches, img_matches); 	
		char str_i[4];
		_itoa_s(ret_matches ,str_i,10);
		putText(img_matches, str_i, cvPoint( 320,20 ),CV_FONT_HERSHEY_COMPLEX,0.7,Scalar(0,255,0));
		_itoa_s(query_keypoint.size(),str_i,10);
		putText(img_matches, str_i, cvPoint( 320,40 ),CV_FONT_HERSHEY_COMPLEX,0.7,Scalar(0,255,0));
		cvWaitKey(50);
		imwrite(file_name, img_matches);
	}
	
	
	/*
	Mat img_matches; 
	if(query_keypoint.size() == 0 || reference_keypoint.size() == 0)
	{
		return good_match_proportion;
	}
	drawMatches(query_img, query_keypoint, reference_img , reference_keypoint, good_matches, img_matches); 	
	char str_i[4];
	_itoa_s(ret_matches ,str_i,10);
	putText(img_matches, str_i, cvPoint( 320,20 ),CV_FONT_HERSHEY_COMPLEX,0.7,Scalar(0,255,0));
	_itoa_s(query_keypoint.size(),str_i,10);
	putText(img_matches, str_i, cvPoint( 320,40 ),CV_FONT_HERSHEY_COMPLEX,0.7,Scalar(0,255,0));
	cvWaitKey(50);
	imshow("Alignment", img_matches);
	*/
	return good_match_proportion;

}

void draw()
{

}

void PersistOnDisk(char* movie_name, string file_name)
{
	cv::VideoCapture capture(movie_name);  

    if (!capture.isOpened()) {  
        cout<<"ERROR"<<endl; 
    }  

	PersistOnDisk(capture, file_name);
}

void PersistOnDisk(VideoCapture capture, string file_name)
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

		if(cur_index % DOWNSAMPLE_RATE != 0)
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
		string absolute_file_name = "F://KEYPOINTS//" + file_name + "//" + ss.str();
		string absolute_keypoint_file_name = "F://KEYPOINTS//" + file_name + "//" + ss.str() + "_KeyPoint";
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

vector<KeyPoint> get_key_point(string file_name)
{
	vector<KeyPoint> mykpts2;	
	string absolute_keypoint_file_name = file_name;
	FileStorage fs2(absolute_keypoint_file_name, FileStorage::READ);
	FileNode kptFileNode = fs2["keypoint"];
	read( kptFileNode, mykpts2 );
	fs2.release();

	return mykpts2;
}

//Most Time-Consuming-Part
int CompareMatWithHistory(Mat query_frame, string file_name, int start_frame, int end_frame, int& max_similar_rate)
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

	//xxxxxxyyy xxx-> Frame Index yyy->matches because yyy<500
	vector<long> result_vec;
	
	int best_matched_img = 0;
	int max_match_number = 0;

	int total_block = (end_frame - start_frame) / DOWNSAMPLE_RATE ;

	Mat* reference_descriptor = new Mat[total_block];
	string *absolute_file_name  = new string[total_block];
	string *absolute_keypoint_file_name = new string[total_block]; 
	int* cur_index = new int[total_block];
	int* similar_size = new int[total_block];
	stringstream* ss = new stringstream[total_block];

	//BottleNeck Part, Use Multi-Core OpenMP
	//Multi-Core Apporoach

	#pragma omp parallel for 
	for(int n_count = 0 ; n_count < total_block ; n_count ++)
	{
		cur_index[n_count] = start_frame + DOWNSAMPLE_RATE* n_count;

		ss[n_count]<<cur_index[n_count];

		absolute_file_name[n_count] = "F://KEYPOINTS//" + file_name + "//" + ss[n_count].str();
		absolute_keypoint_file_name[n_count] = "F://KEYPOINTS//" + file_name + "//" + ss[n_count].str() + "_KeyPoint";

		reference_descriptor[n_count] = descriptor_read(absolute_file_name[n_count]);
		similar_size[n_count] = CompareDescriptorWithDiscriptor(query_descriptor,reference_descriptor[n_count],query_keypoint,get_key_point(absolute_keypoint_file_name[n_count])).size();
		//DEBUG PRINT
		/*
		int cur_size = similar_size[n_count];
		#pragma omp critical 
		{
			if(cur_size > max_match_number)
			{
				cout<<similar_size[n_count]<<" "<<max_match_number<<endl;
				max_match_number =  cur_size;
				best_matched_img = cur_index[n_count];
				//cout<<max_match_number<<" "<<best_matched_img<<endl;
			}
		}
		*/
		//#pragma omp atomic
		#pragma omp critical 
		{
			result_vec.push_back(cur_index[n_count] * 1000 + similar_size[n_count]);
		}
	}

	if(result_vec.size() == 0)
	{
		return 0;
	}

	for(int i = 0 ; i < result_vec.size() ; i++ )
	{
		if( result_vec[i] % 1000 > max_match_number)
		{
			max_match_number = result_vec[i] % 1000;
			best_matched_img  = result_vec[i] / 1000;
		}
	}

	max_similar_rate = max_match_number;
	cout<<"Best Match Number: " << max_match_number<<endl;
	cout<<"Best Match Index : " << best_matched_img<<endl;
 
	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "CompareMatWithHistory : %f seconds\n", duration ); 

	delete[] reference_descriptor;
	delete[] absolute_file_name;
	delete[] absolute_keypoint_file_name;
	delete[] cur_index;
	delete[] similar_size;
	delete[] ss;
	

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

	int query_fps = query_cap.get(CV_CAP_PROP_FPS);
	int reference_fps = reference_cap.get(CV_CAP_PROP_FPS);

	cout<<"DEBUG_1 "<<query_total_frame<<" "<<reference_total_frame<<endl;
	cout<<"DEBUG_2 "<<query_fps<<" "<<reference_fps<<endl;


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
	for(int i = 0 ; i < DOWNSAMPLE_RATE ; i++)
	{		
		stringstream ss;
		ss<<i<<".jpg";

		cout<<"GO TO NUM: "<<i<<" Query Frames(0 - 9)"<<endl;
		//choose the middle frame for example add DOWNSAMLE_RATE SO Alignment could not exceed the border
		long query_begin_frame = query_total_frame / DOWNSAMPLE_RATE * i  + i + DOWNSAMPLE_RATE;
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
		imwrite(ss.str(),croped_frame);	

		raw_frame_index = CompareMatWithHistory(croped_frame, movie_folder_name, 0 , reference_total_frame,good_match_number);

		if(good_match_number < MATCHES_DESCRIPTOR_MIN_LIMIT)
		{
			cout<<"NOT FOUND"<<endl;
			is_frame_in_this_clip[i] = false;
			continue;
		}
		
		is_frame_in_this_clip[i] = true;

		int off_size = AlignmentFrameWithinMovie(query_frame, reference_movie, raw_frame_index);
		//int off_size = 0;
		cout<<"Alignment_Off_Size: "<<off_size<<endl;

		//query_begin_frame 1464 total frame 2928 raw_frame 3940 off_size 1 block 20
		int query_clip_begin_block = BinarySearchBorderLeft(query_cap,reference_cap, query_begin_frame - off_size, query_total_frame, raw_frame_index, movie_folder_name);

		//query_begin_frame 1464 total frame 2928 raw_frame 3940 off_size = 1
		int query_clip_end_block = BinarySearchBorderRight(query_cap,reference_cap, query_begin_frame - off_size, query_total_frame, raw_frame_index, movie_folder_name);

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

		
		//TODO should concat the left/right here
		if( (query_clip_end_block - query_clip_begin_block + 1)  / (query_total_frame / DOWNSAMPLE_RATE + 0.01) >  LONG_MATCHES_IN_QUERY_TERMINATE)
		{
			break;
		}
		
	}

	for(int j = 0 ; j < DOWNSAMPLE_RATE ; j++)
	{
		if(is_frame_in_this_clip[j])
		{
			cout<<j<<" "<<query_block_left_border[j]<<" "<<query_block_right_border[j]<<" GAP:"<<query_block_right_border[j] - query_block_left_border[j]<<endl;
			cout<<"Reference Start"<<reference_start_index[j]<<" Query Start "<< query_start_index[j]<<endl;
		}
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

		const int merge_clip_max_offset_threshold = 3;
		const int very_short_clip_block_threshold = 1;
		const int max_gap_block_between_clips = 10;



		//Step 2 extend left or right border
		for(int k = 0 ; k < DOWNSAMPLE_RATE ; k++)
		{
			if(!is_frame_in_this_clip[k])
			{
				continue;
			}
			//if the block is very short, ignore this block
			if(query_block_right_border[k] - query_block_right_border[k] <= very_short_clip_block_threshold) 
			{
				continue;
			}
			//left border should be extend
			if(query_block_left_border[k] < query_begin_max_block)
			{
				//Validate
				//Temporary disable validation
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
				//Temporary disable validation
				if( abs((query_block_left_border[k] - query_begin_max_block) - (query_start_index[k] - query_start_max_frame) / DOWNSAMPLE_RATE) < merge_clip_max_offset_threshold)
				{
					reference_start_max_frame = reference_start_index[k];
					query_start_max_frame = query_start_index[k];
					query_end_max_block = query_block_right_border[k];
				}
			}
		}

		int max_block_extended = query_end_max_block -	query_begin_max_block;
		

		double validate_result = ValidateTwoMovie(query_cap,reference_cap,query_start_max_frame,reference_start_max_frame,max_block_extended);
		
		finish = clock(); 
		duration = (double)(finish - start) / CLOCKS_PER_SEC;  
		printf( "CompareMovieWithMovie : %f seconds\n", duration ); 

		
		//Temproroy Disable the Validation
		if(1)
		//if(validate_result > VALIDATE_THRESHOLD)
		{
			cout<<"#######################################"<<endl;
			cout<<"#FINAL RESULT"<<":Query_Time_Begin:"<<query_begin_max_block * DOWNSAMPLE_RATE  / FRAME_PER_SECOND <<endl;
			cout<<"#FINAL RESULT"<<":Query_Time_End:"<<query_end_max_block * DOWNSAMPLE_RATE  / FRAME_PER_SECOND<<endl;
			cout<<"#FINAL RESULT"<<":Reference_Time_Begin:"<<reference_start_max_frame   / FRAME_PER_SECOND <<endl;
			cout<<"#FINAL RESULT"<<":Reference_Time_End:"<<(reference_start_max_frame +  query_end_max_block * DOWNSAMPLE_RATE - query_begin_max_block * DOWNSAMPLE_RATE )  / FRAME_PER_SECOND<<endl;
			cout<<"#FINAL RESULT"<<":Total Time:"<<(query_end_max_block - query_begin_max_block) * DOWNSAMPLE_RATE  / FRAME_PER_SECOND <<"SECOND"<<endl;
			cout<<"#######################################"<<endl;
			fout<<query_begin_max_block * DOWNSAMPLE_RATE  / FRAME_PER_SECOND<<","<<query_end_max_block * DOWNSAMPLE_RATE  / FRAME_PER_SECOND<<endl;

			//PlayTwoMovie(query_cap,reference_cap,query_start_max_frame,reference_start_max_frame,max_block_extended);
			
		}
		else
		{
			//Validation Failed
			cout<<"#######################################"<<endl;
			cout<<"#		Validation Failed			 #"<<endl;
			cout<<"#######################################"<<endl;
			fout<<"0,0\n";
		}
	}
	else
	{
		//Not Find
		cout<<"#######################################"<<endl;
		cout<<"#				NOT FOUND			 #"<<endl;
		cout<<"#######################################"<<endl;
		fout<<"0,0\n";
	}

	return 0;
}

double ValidateTwoMovie(VideoCapture query_cap, VideoCapture reference_cap, int start_query, int start_reference, int blocks)
{
	Mat query_mat, reference_mat, concat_mat;

	const int number_of_validate = 10;

	if(blocks < number_of_validate)
	{
		cout<<"Validate Result: So less Blocks"<<endl;
		return 1;
	}
	
	int jump_block = blocks  / number_of_validate;
	int test_cases = 0;
	int good_cases = 0;
	for(int i = jump_block ; i < blocks ; i += jump_block)
	{
		int frame_in_the_reference = start_reference + i * DOWNSAMPLE_RATE;
		int frame_in_the_query = start_query + i * DOWNSAMPLE_RATE;
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

		double good_match_proportion = CompareBrief(query_mat, reference_mat);

		cout<<"Validating Test "<<test_cases<<" :"<<good_match_proportion<<endl;
		if(good_match_proportion > SIMILARITY_THRESHOLD_VALIDATE)
		{
			good_cases ++;
		}
		test_cases++;
	}
	cout<<"Validate Result:"<<"--Good Cases:"<<good_cases<<"--Total Cases:"<<test_cases<<endl;

	return good_cases/(test_cases + 0.01);

}

void PlayTwoMovie(VideoCapture query_cap, VideoCapture reference_cap, int start_query, int start_reference, int blocks)
{
	Mat query_mat, reference_mat, concat_mat;
	for(int i = 0 ; i < blocks ; i++)
	{
		//down_sample
		if(i % DOWNSAMPLE_RATE != 1)
		{
			continue;
		}
		int frame_in_the_reference = start_reference + i * DOWNSAMPLE_RATE;
		int frame_in_the_query = start_query + i * DOWNSAMPLE_RATE;
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
		cvWaitKey(2);
		
	}
}

//2828, 5656, 7880, -5, xxx.mp4
//2828 - 5 = 2323
//2323 = 20 * 116 block
int BinarySearchBorderLeft(VideoCapture query_cap, VideoCapture reference_cap,int aligned_query_begin_frame, int total_query_frame, int raw_reference_frame, char* movie_folder_name)
{
	int total_number_of_block = aligned_query_begin_frame / DOWNSAMPLE_RATE;
	int reference_start = raw_reference_frame - total_number_of_block * DOWNSAMPLE_RATE;
	int alignment_query_start = aligned_query_begin_frame - total_number_of_block * DOWNSAMPLE_RATE;
	//find the first block match the refernece
	int start_block = 0;
	int end_block = aligned_query_begin_frame / DOWNSAMPLE_RATE;
	//Think of the beginning of the movie may be the advertisement, or intro, we use step to get close to the front
	double difference = 0;
	Mat query_mat,reference_mat,concat_mat;
	while(end_block > start_block)
	{
		//Try Half way of the movie
		int cur_block = ceil((start_block + end_block + 0.0) / 2);
		int frame_in_the_reference = reference_start + cur_block * DOWNSAMPLE_RATE;
		int frame_in_the_query = alignment_query_start + cur_block * DOWNSAMPLE_RATE;

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

		double similar_result_cur = 0;
		double similar_result_offset_1 = 0;
		double similar_result_offset_2 = 0;

		//Go to 16:9
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

		else
		{
			resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			similar_result_cur = CompareBrief(query_mat,reference_mat);
		}

		//Find another frame after this block
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query + DOWNSAMPLE_RATE);
		query_cap.read(query_mat);

		reference_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_reference + DOWNSAMPLE_RATE);
		reference_cap.read(reference_mat);
		
		if(query_mat.cols == 0 || query_mat.rows == 0 || reference_mat.cols == 0 || reference_mat.rows == 0)
		{
			similar_result_offset_1 = similar_result_cur; //Bad Frame
		}
		else
		{
			resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			similar_result_offset_1 = CompareBrief(query_mat,reference_mat);
		}

		//Find another frame after 2 blocks
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query + DOWNSAMPLE_RATE * 2);
		query_cap.read(query_mat);

		reference_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_reference + DOWNSAMPLE_RATE * 2);
		reference_cap.read(reference_mat);
		
		if(query_mat.cols == 0 || query_mat.rows == 0 || reference_mat.cols == 0 || reference_mat.rows == 0)
		{
			similar_result_offset_2 = similar_result_cur; //Bad Frame
		}
		else
		{
			resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			similar_result_offset_2 = CompareBrief(query_mat,reference_mat);
		}

		double similar_result = (similar_result_cur + similar_result_offset_1 +similar_result_offset_2) / 3 ;

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
		/*
		//Vertically Concat two Mat
		hconcat(query_mat,reference_mat,concat_mat);
		imshow("Left Border Search",concat_mat);
		cvWaitKey(50);
		imwrite("Left.jpg",concat_mat);
		*/
	}

	return end_block;
}

//2828, 5656, 7880, -5, xxx.mp4
//2828 - 5 = 2323
//2323 = 20 * 116 block
int BinarySearchBorderRight(VideoCapture query_cap, VideoCapture reference_cap,int aligned_query_begin_frame, int total_query_frame, int raw_reference_frame, char* movie_folder_name)
{
	int total_number_of_block = aligned_query_begin_frame / DOWNSAMPLE_RATE;
	int reference_start = raw_reference_frame - total_number_of_block * DOWNSAMPLE_RATE;
	int alignment_query_start = aligned_query_begin_frame - total_number_of_block * DOWNSAMPLE_RATE;


	//find the first block match the refernece
	int start_block = aligned_query_begin_frame / DOWNSAMPLE_RATE;
	int end_block = total_query_frame / DOWNSAMPLE_RATE;
	//Think of the beginning of the movie may be the advertisement, or intro, we use step to get close to the front
	Mat query_mat,reference_mat,concat_mat;

	int reference_frame_total = reference_cap.get(CV_CAP_PROP_FRAME_COUNT);
	while(end_block > start_block)
	{
		//Try Half way of the movie
		int cur_block = ceil((start_block + end_block + 0.0) / 2);
		int frame_in_the_reference = reference_start + cur_block * DOWNSAMPLE_RATE;
		int frame_in_the_query = alignment_query_start + cur_block * DOWNSAMPLE_RATE;

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

		double similar_result_cur = 0;
		double similar_result_offset_1 = 0;
		double similar_result_offset_2 = 0;

		//Go to 16:9
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query);
		query_cap.read(query_mat);

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
		else
		{
			resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			similar_result_cur = CompareBrief(query_mat,reference_mat);
		}

		//Find another frame after this block
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query + DOWNSAMPLE_RATE);
		query_cap.read(query_mat);

		reference_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_reference + DOWNSAMPLE_RATE);
		reference_cap.read(reference_mat);
		
		if(query_mat.cols == 0 || query_mat.rows == 0 || reference_mat.cols == 0 || reference_mat.rows == 0)
		{
			similar_result_offset_1 = similar_result_cur; //Bad Frame
		}
		else
		{
			resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			similar_result_offset_1 = CompareBrief(query_mat,reference_mat);
		}

		//Find another frame after 2 blocks
		query_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_query + DOWNSAMPLE_RATE * 2);
		query_cap.read(query_mat);

		reference_cap.set( CV_CAP_PROP_POS_FRAMES,frame_in_the_reference + DOWNSAMPLE_RATE * 2);
		reference_cap.read(reference_mat);
		
		if(query_mat.cols == 0 || query_mat.rows == 0 || reference_mat.cols == 0 || reference_mat.rows == 0)
		{
			similar_result_offset_2 = similar_result_cur; //Bad Frame
		}
		else
		{
			resize(query_mat,query_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			resize(reference_mat,reference_mat,Size(SHOWOFF_COMPARE_SIZE_X,SHOWOFF_COMPARE_SIZE_Y), 0, 0, 1);
			similar_result_offset_2 = CompareBrief(query_mat,reference_mat);
		}

		double similar_result = (similar_result_cur + similar_result_offset_1 +similar_result_offset_2) / 3 ;

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
			
		/*
		//Vertically Concat two Mat
		hconcat(query_mat,reference_mat,concat_mat);
		imshow("Right Border Search",concat_mat);
		cvWaitKey(50);
		imwrite("Right.jpg",concat_mat);
		*/
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

void preprocessing(char* video_path, char* output_folder_name)
{
	clock_t start, finish;  
	double duration;  
	long i = 10000000L; 
	start = clock();

	PersistOnDisk(video_path,output_folder_name);

	finish = clock(); 
	duration = (double)(finish - start) / CLOCKS_PER_SEC;  
	printf( "preprocessing : %f seconds\n", duration ); 

}

int AlignmentFrameWithinMovie(Mat query_mat, char* movie_path, int raw_frame_index)
{
	cv::VideoCapture reference_cap(movie_path);    
  
    if (!reference_cap.isOpened()) {  
        cout<<"ERROR"<<endl; 
    }  

	long start_frame = (raw_frame_index - DOWNSAMPLE_RATE) >= 0 ? (raw_frame_index - DOWNSAMPLE_RATE) : 0;
	long end_frame   = (raw_frame_index + DOWNSAMPLE_RATE) <= 1000000 ? (raw_frame_index + DOWNSAMPLE_RATE) : CV_CAP_PROP_FRAME_COUNT;
	
	int offsize = 0;
	double best_match_proprotion = 0;
	Mat best_reference_mat;

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
			best_reference_mat = reference_mat;
		}
		//cout<<curFrame<<" "<<good_match_proportion<<endl;
	}

	stringstream ss;
	ss<<start_frame+offsize;

	string file_name = "D:\\Result\\best_match_" + ss.str() + ".jpg";
	CompareBrief(query_mat,best_reference_mat,true,file_name);

	return offsize;
}

void test()
{
	//Jitter
	
	char* ref_test[8] = {"F:\\1.ts","F:\\2.ts","F:\\3.ts","F:\\4.ts","F:\\5.ts","F:\\6.ts","F:\\7.ts","F:\\8.ts"};
	char* fol_test[8] = {"R1","R2","R3","R4","R5","R6","R7","R8"};

	for(int i = 0 ; i < 8 ; i++)
	{
		preprocessing(ref_test[i],fol_test[i]);
		for(int j = 0 ; j < 5 ; j ++)
		{
			stringstream ss;
			ss<<"F:\\Compression\\";
			ss<<"Jitter";
			ss<<"\\";
			ss<<i + 1;
			ss<<".j.";
			ss<<j + 1;
			ss<<".ts";
			string query_movie = ss.str();
			char a[100] = "";
			query_movie.copy(a,90,0);
			cout<<ref_test[i]<<" "<<a<<endl;
			CompareMovieWithMovie(ref_test[i], a,fol_test[i]);
			}
	}
	
	/*
	//Delay
	char* ref_test[8] = {"F:\\1.ts","F:\\2.ts","F:\\3.ts","F:\\4.ts","F:\\5.ts","F:\\6.ts","F:\\7.ts","F:\\8.ts"};
	char* fol_test[8] = {"R1","R2","R3","R4","R5","R6","R7","R8"};
	char* delay_time[7] = {"0.1", "0.4", "1","3","5","8","10"};
	for(int i = 0 ; i < 8 ; i++)
	{
		//preprocessing(ref_test[i],fol_test[i]);
		for(int j = 0 ; j < 7 ; j ++)
		{
			stringstream ss;
			ss<<"F:\\Compression\\";
			ss<<"PLR";
			ss<<"\\";
			ss<<i + 1;
			ss<<".plr.";
			ss<<delay_time[j];
			ss<<".ts";
			string query_movie = ss.str();
			char a[100] = "";
			query_movie.copy(a,90,0);
			cout<<ref_test[i]<<" "<<a<<endl;
			CompareMovieWithMovie(ref_test[i], a,fol_test[i]);
			}
	}
	

	/*
	char* video_path_1 = "F:\\reference_data\\a.mp4";
	char* video_path_2 = "F:\\reference_data\\b.mp4";
	char* video_path_3 = "F:\\reference_data\\c.mp4";
	//preprocessing(video_path_1,"Reference1");
	//preprocessing(video_path_2,"Reference2");
	//preprocessing(video_path_3,"Reference3");
	char* query_clip1 = "F:\\A+B+C\\1280720\\A+B+C.mp4";
	char* query_clip2 = "F:\\A+B+C\\720576\\A+B+C_720576.mp4";
	char* query_clip3 = "F:\\A+B+C\\320240\\A+B+C_320240.mp4";
	char* query_clip4 = "F:\\A+B+C\\720480\\A+B+C_720480.mp4";
	char* query_clip5 = "F:\\A+B+C\\400240\\A+B+C_400240.mp4";
	char* query_clip6 = "F:\\A+B+C\\32\\A+B+C_32.mp4";
	char* query_clip7 = "F:\\A+B+C\\43\\A+B+C_43.mp4";
	char* query_clip8 = "F:\\A+B+C\\54\\A+B+C_54.mp4";

	char* reference_folder_name_1 = "Reference1";
	char* reference_folder_name_2 = "Reference2";
	char* reference_folder_name_3 = "Reference3";

	CompareMovieWithMovie(video_path_1, query_clip3,reference_folder_name_1);
	CompareMovieWithMovie(video_path_2, query_clip3,reference_folder_name_2);
	CompareMovieWithMovie(video_path_3, query_clip3,reference_folder_name_3);
	
	char* reference_folder_name = "Bill";
	char* reference_clip_test = "F:\\core_dataset_30FPS\\core_dataset_FPS30\\bill_clinton_apology_speech\\Clip1_bill.mp4";
	char* query_clip_now_ok = "F:\\core_dataset_30FPS\\core_dataset_FPS30\\bill_clinton_apology_speech\\Clip3_bill.mp4";
	//preprocessing(reference_clip_test,reference_folder_name);
	CompareMovieWithMovie(reference_clip_test, query_clip_now_ok,reference_folder_name);
	*/


	/*
	char* reference_folder_name = "BeatifulMind";
	char* reference_clip_test = "F:\\core_dataset\\core_dataset\\beautiful_mind_game_theory\\Clip2_beautiful.mp4";
	char* query_clip_now_ok =   "F:\\core_dataset\\core_dataset\\beautiful_mind_game_theory\\videoplayback.mp4";
	//preprocessing(reference_clip_test,reference_folder_name);
	CompareMovieWithMovie(reference_clip_test, query_clip_now_ok,reference_folder_name);
	*/

	//char* reference_clip_test = "F:\\core_dataset\\core_dataset\\endless_love\\6834f12ea580a4e7309bd8541fb6fbe4daeeda0f.mp4";
	//char* query_clip_test_1 = "F:\\core_dataset\\core_dataset\\endless_love\\f8b09ea589093ea7fc20cb530963fe610e90fd12.mp4";
	//char* query_clip_test_2 = "F:\\core_dataset\\core_dataset\\endless_love\\5b90735d0a44043685ced4a784dfd3fc8489317a.mp4";
	
	//preprocessing(reference_clip_test,reference_folder_name);
	//CompareMovieWithMovie(reference_clip_test, query_clip_test_1,reference_folder_name);


	/*
	//Test
	string video_path_inida = "F:\\reference_3min\\DTNRD.mp4";
	string video_path_007 = "F:\\reference_3min\\Spy(2015).mp4";
	string video_path_dead = "F:\\reference_3min\\TMJ.mp4";
	//preprocessing(video_path_inida,"Reference1");
	//preprocessing(video_path_007,"Reference2");
	//preprocessing(video_path_dead,"Reference3");


	
	string begin = "F:\\Enhancement\\Luminance\\";
	string q1 = "DTNRD_";
	string q2 = "Spy(2015)_";
	string q3 = "TMJ_";
	string method1 = "40";
	string end = ".mp4";

	string q[3]  = {q1,q2,q3};	
	string reference[3] = {"Reference1", "Reference2", "Reference3"};
	string ref_movie[3] = {video_path_inida, video_path_007, video_path_dead};
	for(int i = 0 ; i < 3 ; i++)
	{
		string query_movie = begin + method1 + "\\"+ q[i] + "Luminance" + method1 + end;
		char a[100] = "";
		char b[100] = "";
		char c[100] = "";
		query_movie.copy(a,90,0);
		ref_movie[i].copy(b,90,0);
		reference[i].copy(c,90,0);
		CompareMovieWithMovie(b,a,c);
	}
	*/ 
		
}

void show()
{
	char *p0 = "0.jpg";
	char *p1 = "1.jpg";
	char *p2 = "2.jpg";
	char *p3 = "3.jpg";
	char *p4 = "4.jpg";
	char *p5 = "5.jpg";
	char *p6 = "6.jpg";
	char *p7 = "7.jpg";
	char *p8 = "8.jpg";
	char *p9 = "9.jpg";
	Mat ma = imread(p0);
	imshow(p0,ma);
	ma = imread(p1);
	imshow(p1,ma);
	ma = imread(p2);
	imshow(p2,ma);
	ma = imread(p3);
	imshow(p3,ma);
	ma = imread(p4);
	imshow(p4,ma);
	ma = imread(p5);
	imshow(p5,ma);
	ma = imread(p6);
	imshow(p6,ma);
	ma = imread(p7);
	imshow(p7,ma);
	ma = imread(p8);
	imshow(p8,ma);
	ma = imread(p9);
	imshow(p9,ma);

	cvWaitKey();
}


void do_demo()
{
	//1 DTNRD 2 SPY 3 TMJ
	char* r[3] = {"Reference1", "Reference2", "Reference3"};
	char* reference; 
	char* query;
	/*
	//BW
	reference = "F:\\reference_3min\\TMJ.mp4";
	query = "F:\\DataSet_updated\\BW\\TMJ_BW.mp4";
	CompareMovieWithMovie(reference,query,r[2]);
	//Blur1
	query = "F:\\DataSet_updated\\blur1\\TMJ_blur1.mp4";
	CompareMovieWithMovie(reference,query,r[2]);
	//Blur2
	query = "F:\\DataSet_updated\\blur2\\TMJ_blur2.mp4";
	CompareMovieWithMovie(reference,query,r[2]);
	//Crop 35%
	reference = "F:\\reference_3min\\Spy(2015).mp4";
	query = "F:\\DataSet_updated\\Cropped\\35%\\Spy(2015)_Cropped35%.mp4";
	CompareMovieWithMovie(reference,query,r[1]);
	//Mirror
	query = "F:\\DataSet_updated\\Mirror\\Spy(2015)_Mirror.mp4";
	CompareMovieWithMovie(reference,query,r[1]);
	//Camera
	reference = "F:\\DataSet_updated\\Cam_record\\VideoRecaptured\\Original\\waterfall_original.avi";
	query = "F:\\DataSet_updated\\Cam_record\\VideoRecaptured\\Recaptured\\waterfall_recaptured.avi";
	CompareMovieWithMovie(reference,query,"Camera");
	
	//Jitter
	reference = "F:\\Compression\\Reference\\4.ts";
	query = "F:\\Compression\\Jitter\\4.j.3.ts";
	CompareMovieWithMovie(reference,query,"R4");
	//Delay
	reference = "F:\\Compression\\Reference\\7.ts";
	query = "F:\\Compression\\Delay\\7.500ms.ts";
	CompareMovieWithMovie(reference,query,"R7");
	//PLR
	reference = "F:\\Compression\\Reference\\3.ts";
	query = "F:\\Compression\\PLR\\3.plr.8.ts";
	CompareMovieWithMovie(reference,query,"R3");
	//Luminous
	reference = "F:\\reference_3min\\DTNRD.mp4";
	query = "F:\\Enhancement\\Luminance\\40\\DTNRD_Luminance40.mp4";
	CompareMovieWithMovie(reference,query,r[0]);
	
	//Contrast
	query = "F:\\Enhancement\\Contrast\\-20\\DTNRD_Contrast-20.mp4";
	CompareMovieWithMovie(reference,query,r[0]);
	*/	
	//Real-Senaries
	reference = "F:\\1.flv";
	query = "F:\\2.flv";
	preprocessing(reference,"REF1");
	CompareMovieWithMovie(reference ,query,"REF1");
}

void do_paper()
{
	Mat x1;
	Mat x2;
	x1 = imread("test2.jpg");
	x2 =imread("test2.png");
	CompareBrief(x1,x2,true,"paper.jpg");
}

void do_test_automation(string input_folder_name)
{
	//Read_Information

	int all_retrived_num = 0;
	int all_ground_truth_num = 0;
	int correct_retrived_num = 0;

	//annotation_input_file_name = "F:\\VideoDataSet_30\\core_dataset\\annotation\\" + input_folder_name + ".txt";
	annotation_input_file_name = "F:\\VideoDataSet_30\\core_dataset\\annotation_5%\\" + input_folder_name + ".txt";

    ifstream fin(annotation_input_file_name);  
	
	fout<<"#################"<<input_folder_name<<"##############"<<endl;

    string s;  
    while( fin >> s ) 
    {     
		vector<string> string_array;
		SplitString(s,string_array,",");

		string str_reference = "F:\\VideoDataSet_30\\core_dataset\\core_dataset\\" + input_folder_name + "\\" + string_array[0];
		string str_query =  "F:\\VideoDataSet_30\\core_dataset\\core_dataset\\" + input_folder_name + "\\" + string_array[1];

		const char * reference = str_reference.data();
		const char * query = str_query.data();

		char ref[256];
		char que[256];
		char ref_fol[256];

		char result[1024];

		strncpy(ref,reference,255);
		strncpy(que,query,255);
		
		string cmd = "md F:\\KEYPOINTS\\" +  string_array[0];
		system(cmd.data());

		strncpy(ref_fol,string_array[0].data(),255);

		string result_string = string_array[0]+","+string_array[1]+","+string_array[4]+","+string_array[5]+",";
		strncpy(result,result_string.data(),1023);
		fout<<result;

		if(string_array[0] == string_array[1])
		{
			preprocessing(ref,ref_fol);
		}

		CompareMovieWithMovie(ref,que,ref_fol);
		
    }
	fout<<"#########################################################"<<endl;
	/*
	char* reference; 
	char* query;
	reference = "F:\\VideoDataSet_30\\core_dataset\\core_dataset\\beautiful_mind_game_theory\\46f2e964ae16f5c27fad70d6849c76616fad7502.flv";
	query =     "F:\\VideoDataSet_30\\core_dataset\\core_dataset\\beautiful_mind_game_theory\\904c8ebf782357ae78ebd205fe3428ad76b975a5.flv";
	preprocessing(reference,"REF");
	CompareMovieWithMovie(reference ,query,"REF");
	*/
}

void do_some_error_detection(char* ref, char* query, char* keypoint)
{

	CompareMovieWithMovie(ref,query,keypoint);
}

int _tmain(int argc, _TCHAR* argv[])
{
	//test();
	//do_demo();
	//do_paper();
	
	/*
	do_test_automation("beautiful_mind_game_theory");
	do_test_automation("bill_clinton_apology_speech");
	do_test_automation("bolt_beijing_100m");
	do_test_automation("brazil_vs_brazil_nike_commercial_2012");
	do_test_automation("david_beckham_lights_the_olympic_torch");
	do_test_automation("dove_evolution_commercial");
	do_test_automation("endless_love");
	do_test_automation("infernal_affairs_1");
	do_test_automation("kennedy_assassination_slow_motion");
	do_test_automation("maradona_hand_of_god");
	do_test_automation("mr_and_mrs_smith_tango");
	do_test_automation("obama_kicks_door");
	do_test_automation("osama_bin_laden_is_dead_obama_speech_at_white_house");
	do_test_automation("president_obama_takes_oath");
	do_test_automation("ronaldinho_ping_pong");
	do_test_automation("run_forrest_fun");
	do_test_automation("saving_private_ryan_omaha_beach");
	do_test_automation("scent_of_woman_tango");
	do_test_automation("the_last_samurai_last_battle");
	do_test_automation("the_legend_of_1900_magic_waltz");
	do_test_automation("the_pursuit_of_happyness_-_job_interview");
	

	do_test_automation("tom_hanks_winning_an_oscar");
	do_test_automation("troy_achilles_and_hector");
	do_test_automation("zidane_headbutt");
	do_test_automation("bolt_beijing_100m");
	do_test_automation("baggio_penalty_1994");
	do_test_automation("beckham_70_yard_goal");
	do_test_automation("titanic_fly_scene");
	do_test_automation("t-mac_13_points_in_35_seconds");
	
	*/

	char *ref = "F:\\VideoDataSet_30\\core_dataset\\core_dataset\\scent_of_woman_tango\\136eff8ef85eee42b2370ad6e444498980ba63a8.mp4";
	char *que = "F:\\VideoDataSet_30\\core_dataset\\core_dataset\\scent_of_woman_tango\\b9d57bdc2d729020825f9617089e1aee6be798fe.flv";
	char *key = "136eff8ef85eee42b2370ad6e444498980ba63a8.mp4";
	do_some_error_detection(ref,que,key);

	cin.get();


	return EXIT_SUCCESS;
}

