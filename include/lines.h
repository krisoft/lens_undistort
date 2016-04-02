#ifndef LINES_H
#define LINES_H

#include <vector>
#include <opencv2/opencv.hpp>


typedef std::vector<cv::Point> Line;
typedef std::vector<Line> Lines;

void draw_line(cv::Mat frame, Line line, cv::Scalar color);
void extract_lines(cv::Mat frame, Lines &lines);

#endif // LINES_H