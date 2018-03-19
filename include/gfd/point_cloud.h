/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   point_cloud.h
 * Author: jpapadak
 *
 * Created on March 16, 2018, 1:42 PM
 */

#ifndef POINT_CLOUD_H
#define POINT_CLOUD_H

#include <cmath>
//#include <tuple>
#include <vector>
//#include <map>
#include <cassert>
#include <iostream>



#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/calib3d.hpp>

#include <boost/make_shared.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

//#include <pcl/sample_consensus/ransac.h>
//#include <pcl/sample_consensus/sac_model_sphere.h>


namespace cv {
    
    
}

namespace gfd {
    template <typename number_t>
    static cv::Point3_<number_t> reproject(cv::Point2i pixel, number_t depth, const cv::Point_<number_t>& focal_length, const cv::Point_<number_t>& image_center) {
        number_t x = depth*(pixel.x - image_center.x)/focal_length.x;
        number_t y = depth*(pixel.y - image_center.y)/focal_length.y;
        return cv::Point3_<number_t>(x, y, depth);
    }
    
    template <typename number_t>
    static std::vector<cv::Point3_<number_t>> reproject(const std::vector<cv::Point2i>& pixels, const cv::Mat_<number_t>& depth_image, const cv::Point_<number_t>& focal_length, const cv::Point_<number_t>& image_center) {
        
        std::vector<cv::Point3_<number_t>> points;
        points.reserve(pixels.size());
        
        for (const cv::Point2i& pixel : pixels) {
            
            const number_t& z = depth_image.at(pixel.y, pixel.x);
            if (not std::isnan(z)) {
                number_t x = z*(pixel.x - image_center.x)/focal_length.x;
                number_t y = z*(pixel.y - image_center.y)/focal_length.y;
                points.emplace_back(x, y, z);
            }
            
        }
        
        return points;
        
    }
    
    template <typename number_t>
    static cv::Mat_<cv::Vec<number_t, 3>> reproject(const cv::Mat_<number_t>& depth_image, const cv::Point_<number_t>& focal_length, const cv::Point_<number_t>& image_center) {
        
        cv::Mat_<cv::Vec<number_t, 3>> points(depth_image.rows, depth_image.cols, cv::traits::Type<cv::Vec<number_t, 3>>::value);
        
        assert(depth_image.isContinuous());
        size_t pixel_y, pixel_x, index;
        
        for (pixel_y = 0; pixel_y < depth_image.rows; ++pixel_y) {
            for (pixel_x = 0; pixel_x < depth_image.cols; ++pixel_x) {
                index = pixel_y*depth_image.cols + pixel_x;
            
                const number_t& z = reinterpret_cast<number_t*>(depth_image.data)[index];
                cv::Vec<number_t, 3>& point = reinterpret_cast<cv::Vec<number_t, 3>*>(points.data)[index];
                
                if (not std::isnan(z)) {
                    point[0] = z*(pixel_x - image_center.x)/focal_length.x;
                    point[1] = z*(pixel_y - image_center.y)/focal_length.y;
                    point[2] = z;
                } else {
                    point[0] = z;
                    point[1] = z;
                    point[2] = z;
                }
                
            }
            
        }
            
        return points;
        
    }
    
    template <typename number_t>
    static cv::Mat_<cv::Vec<number_t, 3>> reprojectParallelized(const cv::Mat_<number_t>& depth_image, const cv::Point_<number_t>& focal_length, const cv::Point_<number_t>& image_center) {
        
        cv::Mat_<cv::Vec<number_t, 3>> points(depth_image.rows, depth_image.cols, cv::traits::Type<cv::Vec<number_t, 3>>::value);
        assert(depth_image.isContinuous());

        points.forEach(
            [&depth_image, &focal_length, &image_center](cv::Vec<number_t, 3>& point, const int* position) -> void {
                size_t pixel_y = position[0];
                size_t pixel_x = position[1];
                size_t index = pixel_y*depth_image.cols + pixel_x;

                const number_t& z = reinterpret_cast<number_t*>(depth_image.data)[index];
                
                if (not std::isnan(z)) {
                    point[0] = z*(pixel_x - image_center.x)/focal_length.x;
                    point[1] = z*(pixel_y - image_center.y)/focal_length.y;
                    point[2] = z;
                } else {
                    point[0] = z;
                    point[1] = z;
                    point[2] = z;
                }       
            }
        );
            
        return points;
        
    }
    
    template <typename number_t>
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectPCL(const std::vector<cv::Point2i>& pixels, 
            const cv::Mat_<number_t>& depth_image, const cv::Point_<number_t>& focal_length, const cv::Point_<number_t>& image_center) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->is_dense = true;
        cloud->reserve(pixels.size());
        
        for (const cv::Point2i& pixel : pixels) {
            
            const number_t& z = depth_image.at(pixel.y, pixel.x);
            if (not std::isnan(z)) {
                number_t x = z*(pixel.x - image_center.x)/focal_length.x;
                number_t y = z*(pixel.y - image_center.y)/focal_length.y;
                cloud->points.emplace_back(x, y, z);
            }
            
        }
        
        return cloud;
        
    }
    
    template <typename number_t>
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectPCL(const cv::Mat_<number_t>& depth_image, const cv::Point_<number_t>& focal_length, const cv::Point_<number_t>& image_center) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->width = depth_image.cols;
        cloud->height = depth_image.rows;
        cloud->is_dense = false;
        cloud->reserve(depth_image.total());
        
        assert(depth_image.isContinuous());
        size_t pixel_y, pixel_x, index;
        
        for (pixel_y = 0; pixel_y < depth_image.rows; ++pixel_y) {
            for (pixel_x = 0; pixel_x < depth_image.cols; ++pixel_x) {
                index = pixel_y*depth_image.cols + pixel_x;
            
                const number_t& z = reinterpret_cast<number_t*>(depth_image.data)[index];
                
                if (not std::isnan(z)) {
                    number_t x = z*(pixel_x - image_center.x)/focal_length.x;
                    number_t y = z*(pixel_y - image_center.y)/focal_length.y;
                    cloud->points.emplace(cloud->begin() + index, x, y, z);
                } else {
                    cloud->points.emplace(cloud->begin() + index, z, z, z);
                }
                
            }
        }
        
        return cloud;
        
    }
    
    template <typename number_t>
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectPCLParallelized(const cv::Mat_<number_t>& depth_image, const cv::Point_<number_t>& focal_length, const cv::Point_<number_t>& image_center) {
        // Probably slower than reprojectPCL due to cloud->resize default construction
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->width = depth_image.cols;
        cloud->height = depth_image.rows;
        cloud->is_dense = false;
        float nan = std::numeric_limits<float>::quiet_NaN();
        cloud->points.resize(depth_image.total(), pcl::PointXYZ(nan, nan, nan));

        depth_image.forEach(
            [&cloud, &focal_length, &image_center](const number_t& z, const int* position) -> void {
                size_t pixel_y = position[0];
                size_t pixel_x = position[1];
                size_t index = pixel_y*cloud->width + pixel_x;
                
                if (not std::isnan(z)) {
                    number_t x = z*(pixel_x - image_center.x)/focal_length.x;
                    number_t y = z*(pixel_y - image_center.y)/focal_length.y;
                    cloud->at(index) = std::move(pcl::PointXYZ(x, y, z));
                };
            }
        );
        
        return cloud;
    }
    
}

#endif /* POINT_CLOUD_H */

