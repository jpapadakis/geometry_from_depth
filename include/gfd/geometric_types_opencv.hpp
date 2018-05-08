/* 
 * File:   geometric_types_opencv.h
 * Author: Andrew Willis, John Papadakis
 *
 * Created on March 19, 2018, 10:57 AM
 */

#ifndef GEOMETRIC_TYPES_OPENCV_H
#define GEOMETRIC_TYPES_OPENCV_H

#include <cmath>
#include <iostream>

#include <boost/shared_ptr.hpp>

#include <opencv2/core.hpp>

namespace cv {
    
    template<typename scalar_t> class Plane3_ : public cv::Point3_<scalar_t> {
        
        static constexpr scalar_t pi = std::acos(-scalar_t(1));
            
    public:

        static constexpr scalar_t ANGLE_THRESHOLD = 5.0;
        static constexpr scalar_t COPLANAR_COS_ANGLE_THRESHOLD = std::cos(ANGLE_THRESHOLD*pi/180.0); // degrees
        static constexpr scalar_t PERPENDICULAR_SIN_ANGLE_THRESHOLD = std::sin(ANGLE_THRESHOLD*pi/180.0); // degrees
        static constexpr scalar_t COPLANAR_COS_EXPECTED_DIHEDRAL_ANGLE = 1.0; // cos(0)

        typedef boost::shared_ptr<Plane3_<scalar_t> > Ptr;
        typedef boost::shared_ptr<const Plane3_<scalar_t> > ConstPtr;

        Plane3_() : d(0), cv::Point3_<scalar_t>() {
        }

        Plane3_(scalar_t _x, scalar_t _y, scalar_t _z, scalar_t _d) : d(_d), cv::Point3_<scalar_t>(_x, _y, _z) {
        }

        Plane3_(const cv::Point3_<scalar_t>& pt1, const cv::Point3_<scalar_t>& pt2,
                const cv::Point3_<scalar_t>& pt3) {
            setCoeffs(pt1, pt2, pt3);
        }	

        Plane3_<scalar_t> clone() {
            return Plane3_<scalar_t>(this->x, this->y, this->z, this->d);
        }

        scalar_t orthogonalDistanceSquared(cv::Point3_<scalar_t> pt) {
            return evaluate(pt)*evaluate(pt);
        }

        scalar_t orthogonalDistanceSigned(cv::Point3_<scalar_t> pt) {
            return evaluate(pt);
        }

        void setCoeffs(scalar_t _x, scalar_t _y, scalar_t _z, scalar_t _d) {
            this->x = _x;
            this->y = _y;
            this->z = _z;
            this->d = _d;
        }

        void scale(scalar_t scalef) {
            this->x *= scalef;
            this->y *= scalef;
            this->z *= scalef;
            this->d *= scalef;
        }

        scalar_t evaluate(const cv::Point3_<scalar_t>& pt) {
            return this->x*pt.x + this->y*pt.y + this->z*pt.z + this->d;
        }

        scalar_t evaluate(scalar_t x, scalar_t y, scalar_t z) {
            return this->x*x + this->y*y + this->z*z + this->d;
        }

        scalar_t evaluateDerivative(int dim) {
            switch (dim) {
                case 1: return this->x;
                    break;
                case 2: return this->y;
                    break;
                case 3: return this->z;
                    break;
                default: throw std::invalid_argument("invalid dimension");
            }

        }

        scalar_t evaluateDerivative(int dim, scalar_t _x, scalar_t _y, scalar_t _z) {
            switch (dim) {
                case 1: return this->x;
                    break;
                case 2: return this->y;
                    break;
                case 3: return this->z;
                    break;
                default: throw std::invalid_argument("invalid dimension");
            }
        }

        scalar_t evaluateDerivative(int dim, const cv::Point3_<scalar_t>& pt) {
            switch (dim) {
                case 1: return this->x;
                    break;
                case 2: return this->y;
                    break;
                case 3: return this->z;
                    break;
                default: throw std::invalid_argument("invalid dimension");
            }
        }

        void setCoeffs(const cv::Point3_<scalar_t>& pt1, const cv::Point3_<scalar_t>& pt2,
                const cv::Point3_<scalar_t>& pt3) {
            this->x = (pt2.y - pt1.y)*(pt3.z - pt1.z) - (pt3.y - pt1.y)*(pt2.z - pt1.z);
            this->y = (pt2.z - pt1.z)*(pt3.x - pt1.x) - (pt3.z - pt1.z)*(pt2.x - pt1.x);
            this->z = (pt2.x - pt1.x)*(pt3.y - pt1.y) - (pt3.x - pt1.x)*(pt2.y - pt1.y);
            this->d = -(this->x*pt1.x + this->y*pt1.y + this->z*pt1.z);
        }

        scalar_t cosDihedralAngle(const Plane3_<scalar_t> test_plane) const {
            return this->x*test_plane.x + this->y*test_plane.y + this->z*test_plane.z;
        }

        scalar_t angleDistance(Plane3_<scalar_t> planeA) const {
            return COPLANAR_COS_EXPECTED_DIHEDRAL_ANGLE - cosDihedralAngle(planeA);
        }

        bool epsilonEquals(Plane3_<scalar_t> planeA, scalar_t eps = COPLANAR_COS_ANGLE_THRESHOLD) const {
            return (COPLANAR_COS_EXPECTED_DIHEDRAL_ANGLE - cosDihedralAngle(planeA)
                    < COPLANAR_COS_ANGLE_THRESHOLD);
        }

        bool epsilonPerpendicular(Plane3_<scalar_t> planeA, scalar_t eps = PERPENDICULAR_SIN_ANGLE_THRESHOLD) const {
            return (std::abs(cosDihedralAngle(planeA)) < eps);
        }

        void interpolate(scalar_t alpha, Plane3_<scalar_t> planeA, Plane3_<scalar_t> planeB,
                cv::Point3_<scalar_t> pt) {
            this->x = alpha*planeA.x + (1 - alpha)*planeB.x;
            this->y = alpha*planeA.x + (1 - alpha)*planeB.x;
            this->z = alpha*planeA.x + (1 - alpha)*planeB.x;
            this->d = -(this->x*pt.x + this->y*pt.y + this->z*pt.z);
        }

        void convertHessianNormalForm() {
            scalar_t normScale = 1.0/std::sqrt(this->x*this->x +
                    this->y*this->y + this->z*this->z);
            scale(normScale);
        }

        friend std::ostream& operator<<(std::ostream& os, const Plane3_<scalar_t>& p) {
            os << "[" << p.x << ", " << p.y << ", " << p.z
                    << ", " << p.d << "]";
            return os;
        }

        cv::Point3_<scalar_t> uvToXYZ(const cv::Point_<scalar_t>& uv) {
            scalar_t threshold = 0.6; // > 1/sqrt(3)
            static cv::Point3_<scalar_t> uVec;
            static cv::Point3_<scalar_t> vVec;
            if (std::abs<scalar_t>(this->x) <= threshold) {
                scalar_t inverse = 1.0/std::sqrt(this->y*this->y + this->z*this->z);
                uVec = cv::Point3_<scalar_t>((scalar_t) 0, inverse*this->z, -inverse*this->y);
            } else if (std::abs<scalar_t>(this->y) <= threshold) {
                scalar_t inverse = 1.0/std::sqrt(this->x*this->x + this->z*this->z);
                uVec = cv::Point3_<scalar_t>(-inverse*this->z, (scalar_t) 0, inverse*this->x);
            } else {
                scalar_t inverse = 1.0/std::sqrt(this->x*this->x + this->y*this->y);
                uVec = cv::Point3_<scalar_t>(inverse*this->y, -inverse*this->x, (scalar_t) 0);
            }
            vVec = uVec.cross(*this);
            cv::Point3_<scalar_t> pt0(-d * this->x, -d * this->y, -d * this->z);
            pt0.x = pt0.x + uv.x * uVec.x + uv.y * vVec.x;
            pt0.y = pt0.y + uv.x * uVec.y + uv.y * vVec.y;
            pt0.z = pt0.z + uv.x * uVec.z + uv.y * vVec.z;
            return pt0;
        }

        cv::Point_<scalar_t> xyzToUV(const cv::Point3_<scalar_t>& p) {
            scalar_t threshold = 0.6; // > 1/sqrt(3)
            cv::Point_<scalar_t> uv;
            static cv::Point3_<scalar_t> uVec;
            static cv::Point3_<scalar_t> vVec;
            if (std::abs(this->x) <= threshold) {
                scalar_t inverse = 1.0 / std::sqrt(this->y*this->y + this->z*this->z);
                uVec = cv::Point3_<scalar_t>((scalar_t) 0.0, inverse*this->z, -inverse*this->y);
            } else if (std::abs(this->y) <= threshold) {
                scalar_t inverse = 1.0 / std::sqrt(this->x*this->x + this->z*this->z);
                uVec = cv::Point3_<scalar_t>(-inverse*this->z, (scalar_t) 0.0, inverse*this->x);
            } else {
                scalar_t inverse = 1.0 / std::sqrt(this->x*this->x + this->y*this->y);
                uVec = cv::Point3_<scalar_t>(inverse*this->y, -inverse*this->x, (scalar_t) 0.0);
            }
            vVec = uVec.cross(*this);
            cv::Point3_<scalar_t> pt0(-d*this->x, -d*this->y, -d*this->z);
            pt0 = p - pt0;
            uv.x = pt0.dot(uVec);
            uv.y = pt0.dot(vVec);
            return uv;
        }

        cv::Point3_<scalar_t> getClosestPointToOrigin() {
            return cv::Point3_<scalar_t>(-this->d*(*this));
        }

        std::string toString() {
            std::ostringstream stringStream;
            stringStream << "(" << this->x << ", " << this->y
                    << ", " << this->z << ", " << this->d << ")";
            return stringStream.str();
        }

        scalar_t d;
    };
    typedef Plane3_<float> Plane3f;
    typedef Plane3_<double> Plane3d;
    typedef Plane3_<int> Plane3i;
    
    template<typename scalar_t> class LabeledPlane3_ : public Plane3_<scalar_t> {
    public:

        typedef boost::shared_ptr<LabeledPlane3_<scalar_t>> Ptr;
        typedef boost::shared_ptr<const LabeledPlane3_<scalar_t>> ConstPtr;

        LabeledPlane3_() : label(0), Plane3_<scalar_t>() {

        };

        LabeledPlane3_(Plane3_<scalar_t> p, int _label) : label(_label), Plane3_<scalar_t>(p.x, p.y, p.z, p.d) {

        };

        LabeledPlane3_(scalar_t x, scalar_t y, scalar_t z, scalar_t d) : label(0),
        Plane3_<scalar_t>(x, y, z, d) {

        };

        LabeledPlane3_(scalar_t x, scalar_t y, scalar_t z, scalar_t d, int _label) : label(_label),
        Plane3_<scalar_t>(x, y, z, d) {

        };

        LabeledPlane3_<scalar_t> clone() {
            LabeledPlane3_<scalar_t> lp;
            lp.x = this->x;
            lp.y = this->y;
            lp.z = this->z;
            lp.d = this->d;
            lp.label = this->label;
            return lp;
        }

        scalar_t distance(scalar_t theta) {
            scalar_t my_theta = cv::fastAtan2(this->y, this->x); // degrees
            return std::abs(my_theta - theta);
        }

        int label;
    };

    class Consensus {
    public:
        size_t inliers, outliers, invalid;

        Consensus() : inliers(0), outliers(0), invalid(0) {
        }

        Consensus(size_t _inliers, size_t _outliers, size_t _invalid) :
        inliers(_inliers), outliers(_outliers), invalid(_invalid) {

        }

        float consensus() const {
            return ((float) inliers) / (inliers + outliers);
        }

        friend std::ostream& operator<<(std::ostream& os, const Consensus& c) {
            os << "[ i=" << c.inliers << ", o=" << c.outliers << ", nan=" << c.invalid << "]";
            return os;
        }
    };

    class FitStatistics : public Consensus {
    public:
        float error_abs = 0;
        float error_sq = 0;
        float noise = 0;
        
        FitStatistics() {
        }
        
        FitStatistics(float _error_abs, float _error_sq, float _noise, 
                size_t _inliers, size_t _outliers, size_t _invalid) : 
            error_abs(_error_abs), error_sq(_error_sq), noise(_noise), 
            Consensus::Consensus(_inliers, _outliers, _invalid) {}
        
        friend std::ostream& operator<<(std::ostream& os, const FitStatistics& stats) {
            os << "[error = " << stats.error_abs << ", error_sq = " << stats.error_sq 
                << ", noise = " << stats.noise << ", i = " << stats.inliers 
                << ", o = " << stats.outliers << ", nan = " << stats.invalid << "]";
            return os;
        }
        
    };
    
    template <typename scalar_t>
    class PlaneWithStats3_ : public LabeledPlane3_<scalar_t> {
    public:
        FitStatistics stats;
        
    };
    typedef PlaneWithStats3_<float> PlaneWithStats3f;
    typedef PlaneWithStats3_<double> PlaneWithStats3d;
    
    class RectWithError : public cv::Rect, public Consensus {
    public:
        float error, noise;

        RectWithError() : cv::Rect(), error(0), noise(0) {
        }

        RectWithError(int _x, int _y, int _width, int _height) : 
                cv::Rect(_x, _y, _width, _height), error(0), noise(0), Consensus() {

        }

        RectWithError(int _x, int _y, int _width, int _height, float _error,
                int _inliers, int _outliers, int _invalid = 0) :
        cv::Rect(_x, _y, _width, _height), error(_error), noise(0),
        Consensus(_inliers, _outliers, _invalid) {

        }

        RectWithError clone() {
            return RectWithError(x, y, width, height, error,
                    inliers, outliers, invalid);
        }

        void clearStatistics() {
            error = inliers = outliers = invalid = 0;
        }

        void getCenter(int& ix, int& iy) {
            ix = x + (width >> 1);
            iy = y + (height >> 1);
        }

        friend std::ostream& operator<<(std::ostream& os, const RectWithError& r) {
            os << "[ x=" << r.x << ", y=" << r.y << ", w=" << r.width
                    << ", h=" << r.height << ", in=" << r.inliers
                    << ", out=" << r.outliers << ", bad=" << r.invalid << ", e=" << r.error << "]";
            return os;
        }

        class ErrorComparator {
        public:

            //            bool operator()(RectWithError* r1, RectWithError* r2) {
            //                return r1->error > r2->error;
            //            }

            bool operator()(RectWithError r1, RectWithError r2) {
                return r1.error > r2.error;
            }
        };

    };

    
}

#endif /* GEOMETRIC_TYPES_OPENCV_H */

