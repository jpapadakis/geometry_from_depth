/* 
 * File:   plane_fit.h
 * Author: Andrew Willis, John Papadakis
 *
 * Created on March 19, 2018, 9:25 AM
 */

#ifndef PLANE_FIT_H
#define PLANE_FIT_H

#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>

#include <opencv2/core.hpp>
#include <gfd/geometric_types_opencv.h>

#include <boost/make_shared.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

namespace gfd {
    
    static constexpr double DEPTH_NOISE_CONSTANT = 1.425e-3;
    
    template <typename number_t>
    number_t getDepthStandardDeviation(const number_t& depth) {
        return DEPTH_NOISE_CONSTANT*depth*depth;
    }
    
    template <typename number_t>
    static cv::Plane3_<number_t> fitExplicitPlaneLeastSquares(const std::vector<cv::Point3_<number_t>>& points) {
        
        cv::Mat _M = cv::Mat::zeros(points.size(), 3, cv::traits::Type<number_t>::value);
        cv::Mat _Z = cv::Mat::zeros(points.size(), 1, cv::traits::Type<number_t>::value);
        number_t* M = _M.ptr<number_t>();
        number_t* Z = _Z.ptr<number_t>();
        
        for (size_t ptIdx = 0; ptIdx < points.size(); ++ptIdx) {
            M[3 * ptIdx] = points[ptIdx].x;
            M[3 * ptIdx + 1] = points[ptIdx].y;
            M[3 * ptIdx + 2] = 1.0f;
            Z[ptIdx] = points[ptIdx].z;
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat _planeCoeffs = _MtM.inv() * _M.t() * _Z;
        number_t* planeCoeffs = _planeCoeffs.ptr<number_t>();
        //std::cout << "E_error=" << _error << std::endl;
        //std::cout << "M = " << M << std::endl;
        //std::cout << "Z = " << Z << std::endl;
        //std::cout << "eplaneCoeffs = " << _planeCoeffs << std::endl;
        cv::Plane3_<number_t> plane3;
        plane3.x = planeCoeffs[0];
        plane3.y = planeCoeffs[1];
        plane3.z = -1;
        plane3.d = planeCoeffs[2];
        number_t normScale = 1.0 / sqrt(plane3.x * plane3.x + plane3.y * plane3.y + plane3.z * plane3.z);
        plane3.scale(normScale);
        //std::cout << "eplaneCoeffs = " << plane3 << std::endl;
        //std::cout.flush();
        
        return plane3;
        
    }
    
    template <typename number_t>
    static cv::PlaneWithStats3_<number_t> fitExplicitPlaneLeastSquaresWithStats(const std::vector<cv::Point3_<number_t>>& points) {
        
        cv::PlaneWithStats3_<number_t> plane3;
        
        cv::Mat _M = cv::Mat::zeros(points.size(), 3, cv::traits::Type<number_t>::value);
        cv::Mat _Z = cv::Mat::zeros(points.size(), 1, cv::traits::Type<number_t>::value);
        number_t* M = _M.ptr<number_t>();
        number_t* Z = _Z.ptr<number_t>();
        
        for (size_t ptIdx = 0; ptIdx < points.size(); ++ptIdx) {
            M[3 * ptIdx] = points[ptIdx].x;
            M[3 * ptIdx + 1] = points[ptIdx].y;
            M[3 * ptIdx + 2] = 1.0f;
            Z[ptIdx] = points[ptIdx].z;
            plane3.stats.noise += getDepthStandardDeviation(points[ptIdx].z);
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat _planeCoeffs = _MtM.inv() * _M.t() * _Z;
        number_t* planeCoeffs = _planeCoeffs.ptr<number_t>();
        cv::Mat _error = _M * _planeCoeffs - _Z;
        //std::cout << "E_error=" << _error << std::endl;
        //std::cout << "M = " << M << std::endl;
        //std::cout << "Z = " << Z << std::endl;
        //std::cout << "eplaneCoeffs = " << _planeCoeffs << std::endl;           
        number_t* _z = _Z.ptr<number_t>();
        number_t* _err = _error.ptr<number_t>();

        for (size_t ptIdx = 0; ptIdx < points.size(); ptIdx++) {
            if (_err[ptIdx] < getDepthStandardDeviation(_z[ptIdx])) {
                plane3.stats.inliers++;
            } else {
                plane3.stats.outliers++;
            }
            plane3.stats.error_abs += std::abs(_err[ptIdx]);
            plane3.stats.error_sq = _err[ptIdx]*_err[ptIdx];
        }
        
        plane3.stats.error_abs /= points.size();
        plane3.stats.error_sq /= points.size();
        plane3.stats.noise /= points.size();
        plane3.x = planeCoeffs[0];
        plane3.y = planeCoeffs[1];
        plane3.z = -1;
        plane3.d = planeCoeffs[2];
        number_t normScale = 1.0 / sqrt(plane3.x * plane3.x + plane3.y * plane3.y + plane3.z * plane3.z);
        plane3.scale(normScale);
        //std::cout << "eplaneCoeffs = " << plane3 << " error = " << error << std::endl;
        //std::cout.flush();
        
        return plane3;
        
    }

    template <typename number_t>
    static cv::Plane3_<number_t> fitImplicitPlaneLeastSquares(const std::vector<cv::Point3_<number_t>>& points) {
        
        cv::Plane3_<number_t> plane3;
        size_t numPoints = points.size();
        cv::Mat _M = cv::Mat::zeros(numPoints, 3, cv::traits::Type<number_t>::value);
        number_t* M = _M.ptr<number_t>();
        cv::Point3f centroid(0, 0, 0);
        for (size_t ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
            centroid.x += points[ptIdx].x;
            centroid.y += points[ptIdx].y;
            centroid.z += points[ptIdx].z;
        }
        centroid.x /= numPoints;
        centroid.y /= numPoints;
        centroid.z /= numPoints;
        for (size_t ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
            M[3 * ptIdx] = points[ptIdx].x - centroid.x;
            M[3 * ptIdx + 1] = points[ptIdx].y - centroid.y;
            M[3 * ptIdx + 2] = points[ptIdx].z - centroid.z;
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat eigVals, eigVecs;
        cv::eigen(_MtM, eigVals, eigVecs);
        cv::Mat _planeCoeffs = eigVecs.row(2).t();
        //std::cout << "E = " << E << std::endl;
        //std::cout << "MtM = " << _MtM << std::endl;          
        //std::cout << "V = " << eigVecs << std::endl;          
        //std::cout << "coeffs = " << _planeCoeffs << std::endl; 
        plane3.x = _planeCoeffs.at<number_t>(0);
        plane3.y = _planeCoeffs.at<number_t>(1);
        plane3.z = _planeCoeffs.at<number_t>(2);
        plane3.d = -(plane3.x * centroid.x + plane3.y * centroid.y + plane3.z * centroid.z);
//        std::cout << "centroid_dist = " << plane3.evaluate(centroid) << std::endl; cv::Mat _D = cv::Mat::ones(numPoints, 1, cv::traits::Type<number_t>::value);
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
    
    template <typename number_t>
    static cv::PlaneWithStats3_<number_t> fitImplicitPlaneLeastSquaresWithStats(const std::vector<cv::Point3_<number_t>>& points) {
        
        cv::PlaneWithStats3_<number_t> plane3;
        size_t numPoints = points.size();
        
        cv::Mat _M = cv::Mat::zeros(numPoints, 3, cv::traits::Type<number_t>::value);
        number_t* M = _M.ptr<number_t>();
        cv::Point3f centroid(0, 0, 0);
        for (size_t ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
            centroid.x += points[ptIdx].x;
            centroid.y += points[ptIdx].y;
            centroid.z += points[ptIdx].z;
        }
        centroid.x /= numPoints;
        centroid.y /= numPoints;
        centroid.z /= numPoints;
        for (size_t ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
            M[3 * ptIdx] = points[ptIdx].x - centroid.x;
            M[3 * ptIdx + 1] = points[ptIdx].y - centroid.y;
            M[3 * ptIdx + 2] = points[ptIdx].z - centroid.z;
            plane3.stats.noise += getDepthStandardDeviation(points[ptIdx].z);
        }
        cv::Mat _MtM = _M.t() * _M;
        cv::Mat eigVals, eigVecs;
        cv::eigen(_MtM, eigVals, eigVecs);
        cv::Mat _planeCoeffs = eigVecs.row(2).t();
        //std::cout << "E = " << E << std::endl;
        //std::cout << "MtM = " << _MtM << std::endl;          
        //std::cout << "V = " << eigVecs << std::endl;          
        //std::cout << "coeffs = " << _planeCoeffs << std::endl; 
        plane3.x = _planeCoeffs.at<number_t>(0);
        plane3.y = _planeCoeffs.at<number_t>(1);
        plane3.z = _planeCoeffs.at<number_t>(2);
        number_t d3 = -(plane3.x * centroid.x + plane3.y * centroid.y + plane3.z * centroid.z);
        plane3.d = d3;
        //std::cout << "centroid_dist = " << plane3.evaluate(centroid) << std::endl;
        cv::Mat _D = cv::Mat::ones(numPoints, 1, cv::traits::Type<number_t>::value);
        _D *= plane3.d;
        for (size_t ptIdx = 0; ptIdx < numPoints; ++ptIdx) {
            M[3 * ptIdx] += centroid.x;
            M[3 * ptIdx + 1] += centroid.y;
            M[3 * ptIdx + 2] += centroid.z;
        }
        cv::Mat _error = _M * _planeCoeffs + _D;
        //std::cout << "plane.d =" << plane3.d << " D=" << _D << std::endl;
        //std::cout << "I_error=" << _error << std::endl;
        //std::cout << "M = " << M << std::endl;  [0.588991, 0.423888, -0.688047, 1.82959]
        //std::cout << "Z = " << Z << std::endl;
        //number_t* _z = _Z.ptr<number_t>();
        number_t* _err = _error.ptr<number_t>();
        
        for (size_t ptIdx = 0; ptIdx < numPoints; ptIdx++) {
            if (_err[ptIdx] < getDepthStandardDeviation(points[ptIdx].z)) {
                plane3.stats.inliers++;
            } else {
                plane3.stats.outliers++;
            }
            plane3.stats.error_abs += std::abs(_err[ptIdx]);
            plane3.stats.error_sq = _err[ptIdx]*_err[ptIdx];
        }
        plane3.stats.error_abs /= numPoints;
        plane3.stats.error_sq /= numPoints;
        plane3.stats.noise /= numPoints;
        plane3.scale((plane3.z > 0) ? -1.0 : 1.0);
        //std::cout << "iplaneCoeffs = " << plane3 << " error = " << error << std::endl;
        //std::cout.flush();

        return plane3;
        
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

}

#endif /* PLANE_FIT_H */

