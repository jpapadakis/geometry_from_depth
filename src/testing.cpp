
#include <gfd/point_cloud.h>
#include <gfd/plane_fit.h>
#include <iostream>

int main(int argc, char** argv) {
    
    cv::Mat1f depth = cv::Mat1f::eye(100, 100);
    cv::Point2f focal_length(500, 500);
    cv::Point2f center(50, 50);
    
    std::cout << gfd::reproject(depth, focal_length, center).at<cv::Vec3f>(99, 99) << "\n";
    std::cout << gfd::reprojectParallelized(depth, focal_length, center).at<cv::Vec3f>(99, 99) << "\n";
    std::cout << gfd::reprojectPCL(depth, focal_length, center)->at(99, 99) << "\n";
    std::cout << gfd::reprojectPCLParallelized(depth, focal_length, center)->at(99, 99) << "\n";

    return 0;
    
}