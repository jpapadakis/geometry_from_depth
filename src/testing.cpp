
#include <gfd/point_cloud.h>
#include <gfd/plane_fit.h>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    
    cv::Mat1f depth = cv::Mat1f::ones(600, 600);
    cv::Point2f focal_length(500, 500);
    cv::Point2f center(50, 50);
    std::vector<cv::Point2i> locations;
    for (int r = 0; r < depth.rows; r++) {
        for (int c = 0; c < depth.cols; c++) {
            locations.emplace_back(r, c);
        }
    }
    
    std::cout << gfd::reproject(depth, focal_length, center).at(100*depth.cols + 100) << "\n";
    std::cout << gfd::reprojectParallelized(depth, focal_length, center).at(100*depth.cols + 100) << "\n";
    std::cout << gfd::reprojectPCL(depth, focal_length, center)->at(100, 100) << "\n";
    std::cout << gfd::reprojectPCLParallelized(depth, focal_length, center)->at(100, 100) << "\n";
    std::cout << gfd::reproject(locations, depth, focal_length, center).at(100*depth.cols + 100) << "\n";
    std::cout << gfd::reprojectPCL(locations, depth, focal_length, center)->at(100*depth.cols + 100) << "\n";
    std::cout << gfd::reprojectPCLParallelized(locations, depth, focal_length, center)->at(100*depth.cols + 100) << "\n";
    std::vector<cv::Point3f> points = gfd::reproject(depth, focal_length, center);
    
    
    
    
    cv::Plane3f plane, plane2;
    cv::PlaneWithStats3f plane3, plane4;
    plane = gfd::fitImplicitPlaneLeastSquares(points);
    plane2 = gfd::fitExplicitPlaneLeastSquares(points);
    plane3 = gfd::fitImplicitPlaneLeastSquaresWithStats(points);
    plane4 = gfd::fitExplicitPlaneLeastSquaresWithStats(points);
    std::cout << "implicit plane fit: " << plane.toString() << "\n";
    std::cout << "explicit plane fit: " << plane2.toString() << "\n";
    std::cout << "implicit plane fit: " << plane3.toString() << ", stats: " << plane3.stats << "\n";
    std::cout << "explicit plane fit: " << plane4.toString() << ", stats: " << plane4.stats << "\n";
    
//    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
//    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
    
    return 0;
    
}