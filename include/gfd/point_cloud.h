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
//#include <vector>
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


namespace gfd {
    
    static cv::Point3f reproject(cv::Point2i pixel, float depth, const cv::Point2f& focal_length, const cv::Point2f& center) {
        const float& fx = focal_length.x;
        const float& fy = focal_length.y;
        const float& cx = center.x;
        const float& cy = center.y;
        
        float x = depth*(pixel.x - cx)/fx;
        float y = depth*(pixel.y - cy)/fy;
        return cv::Point3f(x, y, depth);
    }
    
    static std::vector<cv::Point3f> reproject(const std::vector<cv::Point2i>& pixels, const cv::Mat1f& depth_image, const cv::Point2f& focal_length, const cv::Point2f& center) {
        
        const float& fx = focal_length.x;
        const float& fy = focal_length.y;
        const float& cx = center.x;
        const float& cy = center.y;
        
        std::vector<cv::Point3f> points;
        points.reserve(pixels.size());
        
        for (const cv::Point2i& pixel : pixels) {
            
            const float& z = depth_image.at<float>(pixel.y, pixel.x);
            if (not std::isnan(z)) {
                float x = z*(pixel.x - cx)/fx;
                float y = z*(pixel.y - cy)/fy;
                points.emplace_back(x, y, z);
            }
            
        }
        
        return points;
        
    }
    
    static cv::Mat3f reproject(const cv::Mat1f& depth_image, const cv::Point2f& focal_length, const cv::Point2f& center) {
        
        const float& fx = focal_length.x;
        const float& fy = focal_length.y;
        const float& cx = center.x;
        const float& cy = center.y;
        
        cv::Mat3f points(depth_image.rows, depth_image.cols, CV_32FC3);
        
        assert(depth_image.isContinuous());
        size_t pixel_y, pixel_x, index;
        
        for (pixel_y = 0; pixel_y < depth_image.rows; ++pixel_y) {
            for (pixel_x = 0; pixel_x < depth_image.cols; ++pixel_x) {
                index = pixel_y*depth_image.cols + pixel_x;
            
                const float& z = reinterpret_cast<float*>(depth_image.data)[index];
                cv::Vec3f& point = reinterpret_cast<cv::Vec3f*>(points.data)[index];
                
                if (not std::isnan(z)) {
                    point[0] = z*(pixel_x - cx)/fx;
                    point[1] = z*(pixel_y - cy)/fy;
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
    
    static cv::Mat3f reprojectParallelized(const cv::Mat1f& depth_image, const cv::Point2f& focal_length, const cv::Point2f& center) {
        
        const float& fx = focal_length.x;
        const float& fy = focal_length.y;
        const float& cx = center.x;
        const float& cy = center.y;
        
        cv::Mat points(depth_image.rows, depth_image.cols, CV_32FC3);
        
        assert(depth_image.isContinuous());

        points.forEach<cv::Vec3f>(
            [&depth_image, &fx, &fy, &cx, &cy](cv::Vec3f& point, const int* position) -> void {
                size_t pixel_y = position[0];
                size_t pixel_x = position[1];
                size_t index = pixel_y*depth_image.cols + pixel_x;

                const float& z = reinterpret_cast<float*>(depth_image.data)[index];
                
                if (not std::isnan(z)) {
                    point[0] = z*(pixel_x - cx)/fx;
                    point[1] = z*(pixel_y - cy)/fy;
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
    
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectPCL(const std::vector<cv::Point2i>& pixel_locations, 
            const cv::Mat1f& depth_image, const cv::Point2f& focal_length, const cv::Point2f& image_center) {
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->is_dense = true;
        cloud->reserve(pixel_locations.size());
        
        for (const cv::Point2i& pixel : pixel_locations) {
            
            const float& z = depth_image.at<float>(pixel.y, pixel.x);
            if (not std::isnan(z)) {
                float x = z*(pixel.x - image_center.x)/focal_length.x;
                float y = z*(pixel.y - image_center.y)/focal_length.y;
                cloud->points.emplace_back(x, y, z);
            }
            
        }
        
        return cloud;
        
    }
    
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectPCL(const cv::Mat1f& depth_image, const cv::Point2f& focal_length, const cv::Point2f& image_center) {
        
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
            
                const float& z = reinterpret_cast<float*>(depth_image.data)[index];
                
                if (not std::isnan(z)) {
                    float x = z*(pixel_x - image_center.x)/focal_length.x;
                    float y = z*(pixel_y - image_center.y)/focal_length.y;
                    cloud->points.emplace(cloud->begin() + index, x, y, z);
                } else {
                    cloud->points.emplace(cloud->begin() + index, z, z, z);
                }
                
            }
        }
        
        return cloud;
        
    }
    
    static pcl::PointCloud<pcl::PointXYZ>::Ptr reprojectPCLParallelized(const cv::Mat& depth_image, const cv::Point2f& focal_length, const cv::Point2f& image_center) {
        // Probably slower than reprojectPCL due to cloud->resize default construction
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->width = depth_image.cols;
        cloud->height = depth_image.rows;
        cloud->is_dense = false;
        float nan = std::numeric_limits<float>::quiet_NaN();
        cloud->points.resize(depth_image.total(), pcl::PointXYZ(nan, nan, nan));

        depth_image.forEach<float>(
            [&cloud, &focal_length, &image_center](const float& z, const int* position) -> void {
                size_t pixel_y = position[0];
                size_t pixel_x = position[1];
                size_t index = pixel_y*cloud->width + pixel_x;
                
                if (not std::isnan(z)) {
                    float x = z*(pixel_x - image_center.x)/focal_length.x;
                    float y = z*(pixel_y - image_center.y)/focal_length.y;
                    cloud->at(index) = std::move(pcl::PointXYZ(x, y, z));
                };
            
            }
        );
        
        return cloud;
    }
    
    
    
}

#endif /* POINT_CLOUD_H */

