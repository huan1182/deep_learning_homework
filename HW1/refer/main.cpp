#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include <time.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

class Conv2D{
	public:
		Conv2D(int ic, int oc, int ks, int s) {
			in_channel = ic;
			o_channel = oc;
			kernel_size = ks;
			stride = s;
			for (int i = 0; i < oc; i++) {
				Mat kernel;
				kernel.create(kernel_size, kernel_size, CV_8UC1); 
				// the type for the kernel is float and only has one channel
				cv::randu(kernel, cv::Scalar::all(-2), cv::Scalar::all(2));
				//cout << A << endl;
				//cout << kernel << endl;
				//cv::waitKey();
				// random init the kernel;
				// in PartB is only has one kernel
				kernels.push_back(kernel);
			}
		}
		
		int c_conv(Mat img) {
			int num_ops = 0;
			int img_height = img.rows;
			int img_width = img.cols;
			
			int kernel_height = kernel_size;
			int kernel_width = kernel_size;
			
			int conv_height = (img_height - kernel_height) / stride + 1;
			int conv_width = (img_width - kernel_width) / stride + 1;
			
			cout << conv_height << ", " << conv_width << endl;
			
			Mat conv_img = Mat::zeros(conv_height, conv_width, CV_32FC1);
			
			vector<Mat> channels;
			for (int k = 0; k < kernels.size(); k++) {
				for (int c = 0; c < 3; c++) {
					for (int h = 0; h < conv_height; h++) {
						int h_pos = h * stride;
						//cout << h << endl;
						for (int w = 0; w < conv_width; w++) {
							int w_pos = w * stride;
							//cout << w_pos << endl;
							if (w_pos + kernel_size < img_width && h_pos + kernel_size < img_height) {
								Mat roi(img, Rect(w_pos, h_pos, kernel_size, kernel_size));
								Mat crop = roi.clone();
								vector<Mat> split_crop;
								split(crop, split_crop);
								//cout << "dot product\n";
								double d = kernels[k].dot(split_crop.at(c).clone());
								conv_img.at<float>(h,w) += d;
								num_ops ++;
							}
						}
						//cout << "lines\n";
					}
					//cout << "channels\n";
				}
			}
			//imwrite("c_conv.png", conv_img);
			
			return num_ops;
		}
		
	private:
		int in_channel;
		int o_channel;
		int kernel_size;
		int stride;
		//vector<Mat> kernels;
		vector<Mat> kernels;
		string mode;
		
};



int main(int argc, char *argv[]) {
	//string file_name = "trees"
	string file_name = "mountain";
	ofstream out(file_name + ".txt");
	if (!out.is_open()) cout << "cannot open an out.txt\n";
	
	
	clock_t startTime, endTime;
	int o_channel = 1;
	Mat img = imread(file_name + ".jpg");
	//cvtColor(img, img, CV_BGR2GRAY);
	for (int i = 0; i < 11; i++) {
		startTime = clock();
		Conv2D conv2d(3,o_channel,3,1);
		o_channel *= 2;
		int ops = conv2d.c_conv(img);
		//out << ops << endl;
		//cout << ops << endl;
		endTime = clock();
		//cout << "ops: " << ops << endl;
		cout << "totol time: " << (endTime - startTime) / CLOCKS_PER_SEC << endl;
		out <<  (endTime - startTime) / CLOCKS_PER_SEC << endl;
	}
	out.close();
	
}