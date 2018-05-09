/* 
 * File:   plane_fit.hpp
 * Author: Andrew Willis, John Papadakis
 *
 * Created on March 19, 2018, 9:25 AM
 */

#ifndef PLANE_FITTING_HPP
#define PLANE_FITTING_HPP

#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <gfd/geometric_types_opencv.hpp>

#include <boost/make_shared.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

namespace gfd {
    
    static constexpr double DEPTH_NOISE_CONSTANT = 1.425e-3;
    
    template <typename scalar_t>
    scalar_t getDepthStandardDeviation(const scalar_t& depth) {
        return DEPTH_NOISE_CONSTANT*depth*depth;
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitPlaneExplicitLeastSquares(const scalar_t* points, size_t num_points, size_t stride) {
        
        cv::Mat _M = cv::Mat::zeros(num_points, 3, cv::traits::Type<scalar_t>::value);
        cv::Mat _Z = cv::Mat::zeros(num_points, 1, cv::traits::Type<scalar_t>::value);
        scalar_t* M = _M.ptr<scalar_t>();
        scalar_t* Z = _Z.ptr<scalar_t>();
        
        for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
            size_t pt_begin = stride*ptIdx;
            M[3 * ptIdx] = points[pt_begin];
            M[3 * ptIdx + 1] = points[pt_begin + 1];
            M[3 * ptIdx + 2] = 1.0;
            Z[ptIdx] = points[pt_begin + 2];
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat _planeCoeffs = _MtM.inv() * _M.t() * _Z;
        scalar_t* planeCoeffs = _planeCoeffs.ptr<scalar_t>();
        //std::cout << "E_error=" << _error << std::endl;
        //std::cout << "M = " << M << std::endl;
        //std::cout << "Z = " << Z << std::endl;
        //std::cout << "eplaneCoeffs = " << _planeCoeffs << std::endl;
        cv::Plane3_<scalar_t> plane3;
        plane3.x = planeCoeffs[0];
        plane3.y = planeCoeffs[1];
        plane3.z = -1;
        plane3.d = planeCoeffs[2];
        scalar_t normScale = 1.0 / sqrt(plane3.x * plane3.x + plane3.y * plane3.y + plane3.z * plane3.z);
        plane3.scale(normScale);
        //std::cout << "eplaneCoeffs = " << plane3 << std::endl;
        //std::cout.flush();
        
        return plane3;
        
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitPlaneExplicitLeastSquares(const std::vector<cv::Point3_<scalar_t>>& points) {
        return fitPlaneExplicitLeastSquares(reinterpret_cast<const scalar_t*>(points.data()), points.size(), 3);
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitPlaneExplicitLeastSquares(const Eigen::Matrix<scalar_t, 4, Eigen::Dynamic, Eigen::ColMajor>& points_homogeneous) {
        return fitPlaneExplicitLeastSquares(points_homogeneous.data(), points_homogeneous.cols(), 4);
    }
    
    template <typename scalar_t>
    static cv::PlaneWithStats3_<scalar_t> fitPlaneExplicitLeastSquaresWithStats(const scalar_t* points, size_t num_points, size_t stride) {
        
        cv::PlaneWithStats3_<scalar_t> plane3;
        
        cv::Mat _M = cv::Mat::zeros(num_points, 3, cv::traits::Type<scalar_t>::value);
        cv::Mat _Z = cv::Mat::zeros(num_points, 1, cv::traits::Type<scalar_t>::value);
        scalar_t* M = _M.ptr<scalar_t>();
        scalar_t* Z = _Z.ptr<scalar_t>();
        
        for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
            size_t pt_begin = stride*ptIdx;
            M[3 * ptIdx] = points[pt_begin];
            M[3 * ptIdx + 1] = points[pt_begin + 1];
            M[3 * ptIdx + 2] = 1.0;
            Z[ptIdx] = points[pt_begin + 2];
            plane3.stats.noise += getDepthStandardDeviation(points[pt_begin + 2]);
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat _planeCoeffs = _MtM.inv() * _M.t() * _Z;
        scalar_t* planeCoeffs = _planeCoeffs.ptr<scalar_t>();
        cv::Mat _error = _M * _planeCoeffs - _Z;
        //std::cout << "E_error=" << _error << std::endl;
        //std::cout << "M = " << M << std::endl;
        //std::cout << "Z = " << Z << std::endl;
        //std::cout << "eplaneCoeffs = " << _planeCoeffs << std::endl;           
        scalar_t* _z = _Z.ptr<scalar_t>();
        scalar_t* _err = _error.ptr<scalar_t>();

        for (size_t ptIdx = 0; ptIdx < num_points; ptIdx++) {
            if (_err[ptIdx] < getDepthStandardDeviation(_z[ptIdx])) {
                plane3.stats.inliers++;
            } else {
                plane3.stats.outliers++;
            }
            plane3.stats.error_abs += std::abs(_err[ptIdx]);
            plane3.stats.error_sq = _err[ptIdx]*_err[ptIdx];
        }
        
        plane3.stats.error_abs /= num_points;
        plane3.stats.error_sq /= num_points;
        plane3.stats.noise /= num_points;
        plane3.x = planeCoeffs[0];
        plane3.y = planeCoeffs[1];
        plane3.z = -1;
        plane3.d = planeCoeffs[2];
        scalar_t normScale = 1.0 / sqrt(plane3.x * plane3.x + plane3.y * plane3.y + plane3.z * plane3.z);
        plane3.scale(normScale);
        //std::cout << "eplaneCoeffs = " << plane3 << " error = " << error << std::endl;
        //std::cout.flush();
        
        return plane3;
        
    }
    
    template <typename scalar_t>
    static cv::PlaneWithStats3_<scalar_t> fitPlaneExplicitLeastSquaresWithStats(const std::vector<cv::Point3_<scalar_t>>& points) {
        return fitPlaneExplicitLeastSquaresWithStats(reinterpret_cast<const scalar_t*>(points.data()), points.size(), 3);
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitPlaneExplicitLeastSquaresWithStats(const Eigen::Matrix<scalar_t, 4, Eigen::Dynamic, Eigen::ColMajor>& points_homogeneous) {
        return fitPlaneExplicitLeastSquaresWithStats(points_homogeneous.data(), points_homogeneous.cols(), 4);
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitImplicitPlaneLeastSquares(const scalar_t* points, size_t num_points, size_t stride) {
        
        cv::Plane3_<scalar_t> plane3;
        
        cv::Mat _M = cv::Mat::zeros(num_points, 3, cv::traits::Type<scalar_t>::value);
        scalar_t* M = _M.ptr<scalar_t>();
        cv::Point3f centroid(0, 0, 0);
        for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
            size_t pt_begin = stride*ptIdx;
            centroid.x += points[pt_begin];
            centroid.y += points[pt_begin + 1];
            centroid.z += points[pt_begin + 2];
        }
        centroid.x /= num_points;
        centroid.y /= num_points;
        centroid.z /= num_points;
        for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
            size_t pt_begin = stride*ptIdx;
            M[3 * ptIdx] = points[pt_begin] - centroid.x;
            M[3 * ptIdx + 1] = points[pt_begin + 1] - centroid.y;
            M[3 * ptIdx + 2] = points[pt_begin + 2] - centroid.z;
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat eigVals, eigVecs;
        cv::eigen(_MtM, eigVals, eigVecs);
        cv::Mat _planeCoeffs = eigVecs.row(2).t();
        //std::cout << "E = " << E << std::endl;
        //std::cout << "MtM = " << _MtM << std::endl;          
        //std::cout << "V = " << eigVecs << std::endl;          
        //std::cout << "coeffs = " << _planeCoeffs << std::endl; 
        plane3.x = _planeCoeffs.at<scalar_t>(0);
        plane3.y = _planeCoeffs.at<scalar_t>(1);
        plane3.z = _planeCoeffs.at<scalar_t>(2);
        plane3.d = -(plane3.x * centroid.x + plane3.y * centroid.y + plane3.z * centroid.z);
//        std::cout << "centroid_dist = " << plane3.evaluate(centroid) << std::endl; cv::Mat _D = cv::Mat::ones(numPoints, 1, cv::traits::Type<scalar_t>::value);
//        cv::Mat _D *= plane3.d;
//        for (size_t ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
//            M[3 * ptIdx] += centroid.x;
//            M[3 * ptIdx + 1] += centroid.y;
//            M[3 * ptIdx + 2] += centroid.z;
//        }
        //std::cout << "plane.d =" << plane3.d << " D=" << _D << std::endl;
        //std::cout << "I_error=" << _error << std::endl;
        //std::cout << "M = " << M << std::endl;  [0.588991, 0.423888, -0.688047, 1.82959]
        //std::cout << "Z = " << Z << std::endl;
        
        plane3.scale((plane3.z > 0) ? -1.0 : 1.0);
        //std::cout << "iplaneCoeffs = " << plane3 << " error = " << error << std::endl;
        //std::cout.flush();

        return plane3;
        
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitPlaneImplicitLeastSquares(const std::vector<cv::Point3_<scalar_t>>& points) {
        return fitImplicitPlaneLeastSquares(reinterpret_cast<const scalar_t*>(points.data()), points.size(), 3);
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitPlaneImplicitLeastSquares(const Eigen::Matrix<scalar_t, 4, Eigen::Dynamic, Eigen::ColMajor>& points_homogeneous) {
        return fitPlaneImplicitLeastSquares(points_homogeneous.data(), points_homogeneous.cols(), 4);
    }
    
    template <typename scalar_t>
    static cv::PlaneWithStats3_<scalar_t> fitPlaneImplicitLeastSquaresWithStats(const scalar_t* points, size_t num_points, size_t stride) {
        
        cv::PlaneWithStats3_<scalar_t> plane3;
        
        cv::Mat _M = cv::Mat::zeros(num_points, 3, cv::traits::Type<scalar_t>::value);
        scalar_t* M = _M.ptr<scalar_t>();
        cv::Point3f centroid(0, 0, 0);
        for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
            size_t pt_begin = stride*ptIdx;
            centroid.x += points[pt_begin];
            centroid.y += points[pt_begin + 1];
            centroid.z += points[pt_begin + 2];
        }
        centroid.x /= num_points;
        centroid.y /= num_points;
        centroid.z /= num_points;
        for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
            size_t pt_begin = stride*ptIdx;
            M[3 * ptIdx] = points[pt_begin] - centroid.x;
            M[3 * ptIdx + 1] = points[pt_begin + 1] - centroid.y;
            M[3 * ptIdx + 2] = points[pt_begin + 2] - centroid.z;
            plane3.stats.noise += getDepthStandardDeviation(points[pt_begin + 2]);
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat eigVals, eigVecs;
        cv::eigen(_MtM, eigVals, eigVecs);
        cv::Mat _planeCoeffs = eigVecs.row(2).t();
        //std::cout << "E = " << E << std::endl;
        //std::cout << "MtM = " << _MtM << std::endl;          
        //std::cout << "V = " << eigVecs << std::endl;          
        //std::cout << "coeffs = " << _planeCoeffs << std::endl; 
        plane3.x = _planeCoeffs.at<scalar_t>(0);
        plane3.y = _planeCoeffs.at<scalar_t>(1);
        plane3.z = _planeCoeffs.at<scalar_t>(2);
        scalar_t d3 = -(plane3.x * centroid.x + plane3.y * centroid.y + plane3.z * centroid.z);
        plane3.d = d3;
        //std::cout << "centroid_dist = " << plane3.evaluate(centroid) << std::endl;
        cv::Mat _D = cv::Mat::ones(num_points, 1, cv::traits::Type<scalar_t>::value);
        _D *= plane3.d;
        for (size_t ptIdx = 0; ptIdx < num_points; ++ptIdx) {
            M[3 * ptIdx] += centroid.x;
            M[3 * ptIdx + 1] += centroid.y;
            M[3 * ptIdx + 2] += centroid.z;
        }
        cv::Mat _error = _M * _planeCoeffs + _D;
        //std::cout << "plane.d =" << plane3.d << " D=" << _D << std::endl;
        //std::cout << "I_error=" << _error << std::endl;
        //std::cout << "M = " << M << std::endl;  [0.588991, 0.423888, -0.688047, 1.82959]
        //std::cout << "Z = " << Z << std::endl;
        //scalar_t* _z = _Z.ptr<scalar_t>();
        scalar_t* _err = _error.ptr<scalar_t>();
        
        for (size_t ptIdx = 0; ptIdx < num_points; ptIdx++) {
            if (_err[ptIdx] < getDepthStandardDeviation(points[stride*ptIdx + 2])) {
                plane3.stats.inliers++;
            } else {
                plane3.stats.outliers++;
            }
            plane3.stats.error_abs += std::abs(_err[ptIdx]);
            plane3.stats.error_sq = _err[ptIdx]*_err[ptIdx];
        }
        plane3.stats.error_abs /= num_points;
        plane3.stats.error_sq /= num_points;
        plane3.stats.noise /= num_points;
        plane3.scale((plane3.z > 0) ? -1.0 : 1.0);
        //std::cout << "iplaneCoeffs = " << plane3 << " error = " << error << std::endl;
        //std::cout.flush();

        return plane3;
        
    }
    
    template <typename scalar_t>
    static cv::PlaneWithStats3_<scalar_t> fitPlaneImplicitLeastSquaresWithStats(const std::vector<cv::Point3_<scalar_t>>& points) {
        return fitPlaneImplicitLeastSquaresWithStats(reinterpret_cast<const scalar_t*>(points.data()), points.size(), 3);
    }
    
    template <typename scalar_t>
    static cv::Plane3_<scalar_t> fitPlaneImplicitLeastSquaresWithStats(const Eigen::Matrix<scalar_t, 4, Eigen::Dynamic, Eigen::ColMajor>& points_homogeneous) {
        return fitPlaneImplicitLeastSquaresWithStats(points_homogeneous.data(), points_homogeneous.cols(), 4);
    }
   
    static cv::Plane3f fitPlaneRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr points, double distance_threshold = .01, size_t max_iterations = 1000, bool refine = true) {
        
        pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr plane_model = 
                boost::make_shared<pcl::SampleConsensusModelPlane<pcl::PointXYZ>>(points);
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(plane_model);
        
        ransac.setDistanceThreshold(distance_threshold);
        ransac.setMaxIterations(max_iterations);
        ransac.computeModel();
        
        if (refine) {
            ransac.refineModel();
        }
        
        Eigen::VectorXf coeffs;
        ransac.getModelCoefficients(coeffs);
//        std::vector<int> inlier_indicies;
//        ransac.getInliers(inlier_indicies);
        
        cv::Plane3f plane(coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
        plane.scale((plane.z > 0) ? -1.0 : 1.0);
        
        return plane;
        
    }
    
    static cv::PlaneWithStats3f fitPlaneRANSACWithStats(pcl::PointCloud<pcl::PointXYZ>::Ptr points, double distance_threshold = .01, size_t max_iterations = 1000, bool refine = true) {
        
        pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr plane_model = 
                boost::make_shared<pcl::SampleConsensusModelPlane<pcl::PointXYZ>>(points);
        pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(plane_model);
        
        ransac.setDistanceThreshold(distance_threshold);
        ransac.setMaxIterations(max_iterations);
        ransac.computeModel();
        
        if (refine) {
            ransac.refineModel();
        }
        
        Eigen::VectorXf coeffs;
        ransac.getModelCoefficients(coeffs);
        std::vector<int> inlier_indicies;
        ransac.getInliers(inlier_indicies);
        
        cv::PlaneWithStats3f plane;
        plane.x = coeffs[0];
        plane.y = coeffs[1];
        plane.z = coeffs[2];
        plane.d = coeffs[3];
        plane.scale((plane.z > 0) ? -1.0 : 1.0);
        
        plane.stats.inliers = inlier_indicies.size();
        plane.stats.outliers = points->size() - plane.stats.inliers;
        for (int inlier_index : inlier_indicies) {
            const pcl::PointXYZ& point = points->at(inlier_index);
            float point_error = plane.evaluate(point.x, point.y, point.z);
            plane.stats.error_abs += point_error;
            plane.stats.error_abs += point_error*point_error;
            plane.stats.noise += getDepthStandardDeviation(point.z);
        }
        plane.stats.error_abs /= plane.stats.inliers;
        plane.stats.error_sq /= plane.stats.inliers;
        plane.stats.noise /= plane.stats.inliers;
        
        return plane;
        
    }
    
    static std::vector<cv::Plane3f> planeExtractionRANSAC(pcl::PointCloud<pcl::PointXYZ>::Ptr points, double distance_threshold = .01, size_t max_iterations = 1000, bool refine = true) {

        
        
        pcl::ModelCoefficients::Ptr coefficients = boost::make_shared<pcl::ModelCoefficients>();
        pcl::PointIndices::Ptr inliers = boost::make_shared<pcl::PointIndices>();
        pcl::SACSegmentation<pcl::PointXYZ> segmentation;
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        
        segmentation.setModelType(pcl::SACMODEL_PLANE);
        segmentation.setMethodType(pcl::SAC_RANSAC);
        segmentation.setDistanceThreshold(distance_threshold);
        segmentation.setOptimizeCoefficients(refine);
        segmentation.setMaxIterations(max_iterations);

        size_t original_size = points->size();
        size_t num_planes = 0;
        float min_percentage = .10;
        std::vector<cv::Plane3f> planes;
        
        while (points->size() > .3*original_size) {
            
            segmentation.setInputCloud(points);
            segmentation.segment(*inliers, *coefficients);
            
            if (inliers->indices.size() < min_percentage*original_size) {
                break;
            }
            
            planes.emplace_back(coefficients->values[0], coefficients->values[1], 
                    coefficients->values[2], coefficients->values[3]);
            cv::Plane3f& plane = planes.back();
            plane.scale((plane.z > 0) ? -1.0 : 1.0);
            
            std::cout << "Fit plane: " << plane << ", percentage of points: " << float(inliers->indices.size())/original_size << std::endl;

            // Extract inliers
            extract.setInputCloud(points);
            extract.setIndices(inliers);
            extract.setNegative(true);
            pcl::PointCloud<pcl::PointXYZ> filtered_points;
            extract.filter(filtered_points); // result contains the outliers
            points->swap(filtered_points);
            
            num_planes++;
        }
    
        return planes;
        
    }

}

#endif /* PLANE_FITTING_HPP */

