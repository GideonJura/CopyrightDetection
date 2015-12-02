#ifndef _PLOT_H
#define _PLOT_H
#include <cmath>
#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <vector>
#include <stdio.h>
using namespace cv;
using namespace std;


class Plot
{
	private:	
	void DrawAxis (IplImage *image);
	void DrawData (IplImage *image);
	int window_height;
	int window_width;


	vector< vector<CvPoint2D64f> >dataset;	
	vector<char> lineTypeSet;
	
	//color
	CvScalar backgroud_color;
	CvScalar axis_color;
	CvScalar text_color;

	
	public:
	IplImage* Figure;

	
	// manual or automatic range
	bool custom_range_y;
	double y_max;
	double y_min;

	double y_scale;

	bool custom_range_x;
	double x_max;
	double x_min;

	double x_scale;
	
	//边界大小
	int border_size;
		
	template<class T>
	void plot(IplImage* image, T *y, size_t Cnt, CvScalar color, char lineType='l');	
	template<class T>
	void plot(T *x, T *y, size_t Cnt, CvScalar color, char lineType='l');
		
	void xlabel(string xlabel_name, CvScalar label_color);
	void ylabel(string ylabel_name, CvScalar label_color);
	//清空图片上的数据
	void clear();
	void title(string title_name);
	
	Plot();
	~Plot();
		
};


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//采用范型设计，因此将实现部分和声明部分放在一个文件中
Plot::Plot()
{
	this->border_size = 30;
	this->window_height = 600;
	this->window_width = 600;;
	this->Figure = cvCreateImage(cvSize(this->window_height, this->window_width),IPL_DEPTH_8U, 3);
	memset(Figure->imageData, 255, sizeof(unsigned char)*Figure->widthStep*Figure->height);
	
	//color
	this->backgroud_color = CV_RGB(255,255,255);
	this->axis_color = CV_RGB(0,0,0);
	this->text_color = CV_RGB(255,0 ,0);
}

Plot::~Plot()
{

}

//范型设计
template<class T>
void Plot::plot(T *X, T *Y, size_t Cnt, CvScalar color, char lineType)
{
	//对数据进行存�?
	T tempX, tempY;
	vector<CvPoint2D64f>data;
	for(int i = 0; i < Cnt;i++)
	{
		tempX = X[i];
		tempY = Y[i];
		data.push_back(cvPoint2D64f((double)tempX, (double)tempY));
	}
	this->dataset.push_back(data);
	this->lineTypeSet.push_back(lineType);
	
	//printf("data count:%d\n", this->dataset.size());
	
	this->DrawData(this->Figure);
}

//TODO
template<class T>
void Plot::plot(IplImage* image, T *Y, size_t Cnt, CvScalar color, char lineType)
{
	//对数据进行存�?
	T tempX, tempY;
	vector<CvPoint2D64f>data;
	for(int i = 0; i < Cnt;i++)
	{
		tempX = i;
		tempY = Y[i];
		data.push_back(cvPoint2D64f((double)tempX, (double)tempY));
	}
	this->dataset.push_back(data);
	this->lineTypeSet.push_back(lineType);
	//printf("data count:%d\n", this->dataset.size());
	this->DrawData(this->Figure);
}

void Plot::clear()
{
	this->dataset.clear();
	//memset(Figure->imageData, 255, sizeof(unsigned char)*Figure->widthStep*Figure->height);
}
void Plot::DrawAxis (IplImage *image)
{

	CvScalar axis_color = this->axis_color;
	CvScalar text_color = this->axis_color;
	
	int bs = this->border_size;		
	int h = this->window_height;
	int w = this->window_width;

	// size of graph
	int gh = h - bs * 2;
	int gw = w - bs * 2;

	// draw the horizontal and vertical axis
	// let x, y axies cross at zero if possible.
	double y_ref = this->y_min;
	if ((this->y_max > 0) && (this->y_min <= 0))
		y_ref = 0;

	int x_axis_pos = h - bs - cvRound((y_ref - this->y_min) * this->y_scale);

	cvLine(image, cvPoint(bs,     x_axis_pos), 
		           cvPoint(w - bs, x_axis_pos),
				   axis_color);
	cvLine(image, cvPoint(bs, h - bs), 
		           cvPoint(bs, h - bs - gh),
				   axis_color);

	// Write the scale of the y axis
	CvFont font;
	cvInitFont(&font,1,0.55,0.7, 0,1,16);

	int chw = 6, chh = 10;
	char text[16];

	// y max
	if ((this->y_max - y_ref) > 0.05 * (this->y_max - this->y_min))
	{
		_snprintf_s(text, sizeof(text)-1, "%.1f", this->y_max);
		cvPutText(image, text, cvPoint(bs / 5, bs - chh / 2), &font, text_color);
	}
	// y min
	if ((y_ref - this->y_min) > 0.05 * (this->y_max - this->y_min))
	{
		_snprintf_s(text, sizeof(text)-1, "%.1f", this->y_min);
		cvPutText(image, text, cvPoint(bs / 5, h - bs + chh), &font, text_color);
	}

	// x axis
	_snprintf_s(text, sizeof(text)-1, "%.1f", y_ref);
	cvPutText(image, text, cvPoint(bs / 5, x_axis_pos + chh / 2), &font, text_color);

	// Write the scale of the x axis
	_snprintf_s(text, sizeof(text)-1, "%.0f", this->x_max );
	cvPutText(image, text, cvPoint(w - bs - strlen(text) * chw, x_axis_pos + chh), 
		      &font, text_color);

	// x min
	_snprintf_s(text, sizeof(text)-1, "%.0f", this->x_min );
	cvPutText(image, text, cvPoint(bs, x_axis_pos + chh), 
		      &font, text_color);
}

//添加对线型的支持
//TODO线型未补充完�?
//标记		线型
//l          直线	
//*          �?
//.          �?
//o          �?
//x          �?
//+          十字 
//s          方块 
//d          菱形 
void Plot::DrawData (IplImage *image)
{
	this->x_min = this->x_max = this->dataset[0][0].x;
	this->y_min = this->y_max = this->dataset[0][0].y;
	
	int bs = this->border_size;
	for(size_t i = 0; i < this->dataset.size(); i++)
	{
		for(size_t j = 0; j < this->dataset[i].size(); j++)
		{
			if(this->dataset[i][j].x < this->x_min)
			{
				this->x_min = this->dataset[i][j].x;
			}else if(this->dataset[i][j].x > this->x_max)
			{
				this->x_max = this->dataset[i][j].x;
			}
		
			if(this->dataset[i][j].y < this->y_min)
			{
				this->y_min = this->dataset[i][j].y;
			}else if(this->dataset[i][j].y > this->y_max)
			{
				this->y_max = this->dataset[i][j].y;
			}
		}
	}
	double x_range = this->x_max - this->x_min;
	double y_range = this->y_max - this->y_min;
	this->x_scale = (image->width - bs*2)/x_range;
	this->y_scale = (image->height- bs*2)/y_range;
	
	
	//清屏
	memset(image->imageData, 255, sizeof(unsigned char)*Figure->widthStep*Figure->height);
	this->DrawAxis(image);
	
	//printf("x_range: %f y_range: %f\n", x_range, y_range);
	//绘制�?
	double tempX, tempY;
	CvPoint prev_point, current_point;
	int radius = 3;
	int slope_radius = (int)(radius*1.414/2 + 0.5);
	for(size_t i = 0; i < this->dataset.size(); i++)
	{
		//printf("dataset[i].size(): %d\n", dataset[i].size());	
		for(size_t j = 0; j < this->dataset[i].size(); j++)
		{
			tempX = (int)((this->dataset[i][j].x - this->x_min)*this->x_scale);
			tempY = (int)((this->dataset[i][j].y - this->y_min)*this->y_scale);
			current_point = cvPoint(bs + tempX, image->height - (tempY + bs));
			
			if(this->lineTypeSet[i] == 'l')
			{
				// draw a line between two points
				if (j >= 1)
				{
					cvLine(image, prev_point, current_point, CV_RGB(255,0,0), 1, CV_AA);
				}		
				prev_point = current_point;
			}else if(this->lineTypeSet[i] == '.')
			{
				cvCircle(image, current_point, 1, this->text_color, -1, 8);
			}else if(this->lineTypeSet[i] == '*')
			{
				
			}else if(this->lineTypeSet[i] == 'o')
			{
				cvCircle(image, current_point, radius, this->text_color, 1, CV_AA);
			}else if(this->lineTypeSet[i] == 'x')
			{
				cvLine(image, cvPoint(current_point.x - slope_radius, current_point.y - slope_radius), 
					   cvPoint(current_point.x + slope_radius, current_point.y + slope_radius), CV_RGB(255,0,0), 1, 8);
					   
				cvLine(image, cvPoint(current_point.x - slope_radius, current_point.y + slope_radius), 
					   cvPoint(current_point.x + slope_radius, current_point.y - slope_radius), CV_RGB(255,0,0), 1, 8);
			}else if(this->lineTypeSet[i] == '+')
			{
				cvLine(image, cvPoint(current_point.x - radius, current_point.y), 
					   cvPoint(current_point.x + radius, current_point.y), CV_RGB(255,0,0), 1, 8);
					   
				cvLine(image, cvPoint(current_point.x, current_point.y - radius), 
					   cvPoint(current_point.x, current_point.y + radius), CV_RGB(255,0,0), 1, 8);	   
			}else if(this->lineTypeSet[i] == 's')
			{
				cvRectangle(image, cvPoint(current_point.x - slope_radius, current_point.y - slope_radius), 
					   cvPoint(current_point.x + slope_radius, current_point.y + slope_radius), CV_RGB(255,0,0), 1, 8);
			}else if(this->lineTypeSet[i] == 'd')
			{
		
			}
				
		}
	}	
}

#endif
