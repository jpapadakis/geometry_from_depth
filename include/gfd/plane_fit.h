/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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

namespace gfd {
    
    template <typename number_t>
    static void fitExplicitPlaneLeastSquares(const std::vector<cv::Point3_<number_t>>& points, cv::Plane3_<number_t>& plane3, number_t& error,
                    number_t& noise, size_t& inliers, size_t& outliers, size_t& invalid){
        
        cv::Mat _M = cv::Mat::zeros(points.size(), 3, cv::traits::Type<number_t>::value);
        cv::Mat _Z = cv::Mat::zeros(points.size(), 1, cv::traits::Type<number_t>::value);
        number_t* M = _M.ptr<number_t>();
        number_t* Z = _Z.ptr<number_t>();
        noise = 0;
        for (size_t ptIdx = 0; ptIdx < points.size(); ++ptIdx) {
            M[3 * ptIdx] = points[ptIdx].x;
            M[3 * ptIdx + 1] = points[ptIdx].y;
            M[3 * ptIdx + 2] = 1.0f;
            Z[ptIdx] = points[ptIdx].z;
            //noise += getErrorStandardDeviation(data3[ptIdx].z);
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
        error = inliers = outliers = invalid = 0;
        //        for (size_t ptIdx = 0; ptIdx < numPoints; ptIdx++) {
        //            if (isInlier(_err[ptIdx], _z[ptIdx])) {
        //                inliers++;
        //            } else {
        //                outliers++;
        //            }
        //            error += abs(_error.at<number_t>(ptIdx));
        //            //error += abs(_z[ptIdx]);
        //        }
        //            error /= numPoints;
        //            noise /= numPoints;
        plane3.x = planeCoeffs[0];
        plane3.y = planeCoeffs[1];
        plane3.z = -1;
        plane3.d = planeCoeffs[2];
        number_t normScale = 1.0 / sqrt(plane3.x * plane3.x + plane3.y * plane3.y + plane3.z * plane3.z);
        plane3.scale(normScale);
        //std::cout << "eplaneCoeffs = " << plane3 << " error = " << error << std::endl;
        //std::cout.flush();
        
        
    }

    template <typename number_t>
    static void fitImplicitPlaneLeastSquares(const std::vector<cv::Point3_<number_t>>& points, cv::Plane3_<number_t>& plane3, number_t& error,
            number_t& noise, size_t& inliers, size_t& outliers, size_t& invalid) {
        
        size_t numPoints = points.size();
        noise = 0;
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
            //noise += getErrorStandardDeviation(data3[ptIdx].z);
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
        error = inliers = outliers = invalid = 0;
        //        for (size_t ptIdx = 0; ptIdx < numPoints; ptIdx++) {
        //            if (isInlier(_err[ptIdx], data3[ptIdx].z)) {
        //                inliers++;
        //            } else {
        //                outliers++;
        //            }
        //            //                error += abs(_error.at<number_t>(ptIdx));
        //            error += abs(_err[ptIdx]);
        //        }
        //            error /= numPoints;
        //            noise /= numPoints;
        plane3.scale((plane3.z > 0) ? -1.0 : 1.0);
        //std::cout << "iplaneCoeffs = " << plane3 << " error = " << error << std::endl;
        //std::cout.flush();

        
        
    }

    
    
}

#endif /* PLANE_FIT_H */

