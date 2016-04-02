#include "lines.h"

#include <cmath>

#include <opencv2/opencv.hpp>


enum EdgeLabel {
	NOT_EDGE,
	TOP,
	BOTTOM,
	LEFT,
	RIGHT
};

EdgeLabel label_point( cv::Point point, cv::Size frame_size )
{
	int edge_width = 4;
	if( point.x<edge_width )
	{
		return EdgeLabel::LEFT;
	}
	if( point.x>frame_size.width-edge_width)
	{
		return EdgeLabel::RIGHT;
	}
	if( point.y<edge_width)
	{
		return EdgeLabel::TOP;
	}
	if( point.y>frame_size.height-edge_width)
	{
		return EdgeLabel::BOTTOM;
	}
	return EdgeLabel::NOT_EDGE;
}


/*
	This splits up contour into pieces. A valid piece runs from one edge to an other, and points near the edges are discarded.
	The splits are stored into split_contours.
	We need frame_size to judge if a point is near the edge or not.
*/
void split_contour(Line contour, Lines &split_contours, cv::Size frame_size )
{
	int line_min_distance = 4;
	// find one point which is on the edge
	int edge_idx = -1;
	for(int i=0; i<contour.size(); i++)
	{
		if( label_point( contour[i], frame_size) != EdgeLabel::NOT_EDGE )
		{
			edge_idx = i;
			break;
		}
	}

	if( edge_idx==-1 )
	{
		// there is no such a point, no splits from here then
		return;
	}

	EdgeLabel last_label = label_point( contour[edge_idx], frame_size );
	EdgeLabel line_from = EdgeLabel::NOT_EDGE; // EdgeLabel::NOT_EDGE is not a valid line_from, it means no line has started yet
	Line split;
	for(int idx = edge_idx+1; idx<edge_idx + contour.size()+1; idx++)
	{
		cv::Point point = contour.at( idx % contour.size() );

		EdgeLabel next_label = label_point( point, frame_size );

		if( last_label==EdgeLabel::NOT_EDGE && next_label!=EdgeLabel::NOT_EDGE )
		{
			split.push_back( point );
			if( split.size()>line_min_distance && line_from!=EdgeLabel::NOT_EDGE && line_from!=next_label )
			{
				split_contours.push_back( split );
			}
			split.clear();
			line_from = EdgeLabel::NOT_EDGE;
		}
		else if ( last_label!=EdgeLabel::NOT_EDGE && next_label==EdgeLabel::NOT_EDGE )
		{
			split.push_back( point );
			line_from = last_label;
		}
		else if ( last_label==EdgeLabel::NOT_EDGE and next_label==EdgeLabel::NOT_EDGE )
		{
			split.push_back( point );
		}

		last_label = next_label;
	}
}

bool good_contrast( Line line, const cv::Mat &frame )
{
	double min_contrast = 20.0;


	double sum_left = 0.0;
	int count_left = 0;
	double sum_right = 0.0;
	int count_right = 0;

	int w = frame.size().width;
	int h = frame.size().height;

	for( int idx=1; idx<line.size(); idx++ )
	{
		cv::Point last_point = line.at(idx-1);
		cv::Point point = line.at(idx);

		double dx = (double)( point.x - last_point.x );
		double dy = (double)( point.y - last_point.y );

		double s = 2.0 / std::sqrt( dx*dx + dy*dy );

		int lx = (int)( point.x + dy * s );
		int ly = (int)( point.y - dx * s );

		if ( lx >= 0 && lx < w && ly >= 0 && ly < h )
		{
			count_left += 1;
			sum_left += frame.at<uint8_t>(cv::Point(lx,ly));
		}

		int rx = (int)( point.x - dy * s );
		int ry = (int)( point.y + dx * s );

		if ( rx >= 0 && rx < w && ry >= 0 && ry < h )
		{
			count_right += 1;
			sum_right += frame.at<uint8_t>(cv::Point(rx,ry));
		}
	}

	if ( count_left == 0 || count_right == 0 )
	{
		return false;
	}

	double average_left = sum_left / count_left;
	double average_right = sum_right / count_right;
	return std::abs(average_left-average_right)>min_contrast;
}

void extract_lines(cv::Mat frame, Lines &lines)
{
	cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);

	cv::Mat smoothed_frame;
	cv::medianBlur(frame, smoothed_frame, 15);

	cv::Mat thres;
	cv::adaptiveThreshold(smoothed_frame, thres, 
		255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2
	);

	Lines contours;
  	std::vector<cv::Vec4i> hierarchy;

	cv::findContours( thres, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );

	Lines split_contours;
	for(Line &contour : contours)
	{
		split_contour(contour, split_contours, frame.size() );
	}


	for( Line &split : split_contours )
	{
		if ( good_contrast( split, frame ) )
		{
			lines.push_back( split );
		}
	}
}


void draw_line(cv::Mat frame, Line line, cv::Scalar color)
{
	for( int idx=1; idx<line.size(); idx++ )
	{
		cv::line( frame, line.at(idx-1), line.at(idx), color, 1);
	}
}