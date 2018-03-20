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
    
    template<typename number_t> class Plane3_ : public cv::Point3_<number_t> {
        
        static constexpr number_t pi = std::acos(-number_t(1));
            
    public:

        static constexpr number_t ANGLE_THRESHOLD = 5.0;
        static constexpr number_t COPLANAR_COS_ANGLE_THRESHOLD = std::cos(ANGLE_THRESHOLD*pi/180.0); // degrees
        static constexpr number_t PERPENDICULAR_SIN_ANGLE_THRESHOLD = std::sin(ANGLE_THRESHOLD*pi/180.0); // degrees
        static constexpr number_t COPLANAR_COS_EXPECTED_DIHEDRAL_ANGLE = 1.0; // cos(0)

        typedef boost::shared_ptr<Plane3_<number_t> > Ptr;
        typedef boost::shared_ptr<const Plane3_<number_t> > ConstPtr;

        Plane3_() : d(0), cv::Point3_<number_t>() {
        }

        Plane3_(number_t _x, number_t _y, number_t _z, number_t _d) : d(_d), cv::Point3_<number_t>(_x, _y, _z) {
        }

        Plane3_(const cv::Point3_<number_t>& pt1, const cv::Point3_<number_t>& pt2,
                const cv::Point3_<number_t>& pt3) {
            setCoeffs(pt1, pt2, pt3);
        }	

        Plane3_<number_t> clone() {
            return Plane3_<number_t>(this->x, this->y, this->z, this->d);
        }

        number_t orthogonalDistanceSquared(cv::Point3_<number_t> pt) {
            return evaluate(pt)*evaluate(pt);
        }

        number_t orthogonalDistanceSigned(cv::Point3_<number_t> pt) {
            return evaluate(pt);
        }

        void setCoeffs(number_t _x, number_t _y, number_t _z, number_t _d) {
            this->x = _x;
            this->y = _y;
            this->z = _z;
            this->d = _d;
        }

        void scale(number_t scalef) {
            this->x *= scalef;
            this->y *= scalef;
            this->z *= scalef;
            this->d *= scalef;
        }

        number_t evaluate(const cv::Point3_<number_t>& pt) {
            return this->x * pt.x + this->y * pt.y + this->z * pt.z + this->d;
        }

        number_t evaluate(number_t _x, number_t _y, number_t _z) {
            return this->x * _x + this->y * _y + this->z * _z + this->d;
        }

        number_t evaluateDerivative(int dim) {
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

        number_t evaluateDerivative(int dim, number_t _x, number_t _y, number_t _z) {
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

        number_t evaluateDerivative(int dim, const cv::Point3_<number_t>& pt) {
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

        void setCoeffs(const cv::Point3_<number_t>& pt1, const cv::Point3_<number_t>& pt2,
                const cv::Point3_<number_t>& pt3) {
            this->x = (pt2.y - pt1.y)*(pt3.z - pt1.z) - (pt3.y - pt1.y)*(pt2.z - pt1.z);
            this->y = (pt2.z - pt1.z)*(pt3.x - pt1.x) - (pt3.z - pt1.z)*(pt2.x - pt1.x);
            this->z = (pt2.x - pt1.x)*(pt3.y - pt1.y) - (pt3.x - pt1.x)*(pt2.y - pt1.y);
            this->d = -(this->x*pt1.x + this->y*pt1.y + this->z*pt1.z);
        }

        number_t cosDihedralAngle(const Plane3_<number_t> test_plane) const {
            return this->x*test_plane.x + this->y*test_plane.y + this->z*test_plane.z;
        }

        number_t angleDistance(Plane3_<number_t> planeA) const {
            return COPLANAR_COS_EXPECTED_DIHEDRAL_ANGLE - cosDihedralAngle(planeA);
        }

        bool epsilonEquals(Plane3_<number_t> planeA, number_t eps = COPLANAR_COS_ANGLE_THRESHOLD) const {
            return (COPLANAR_COS_EXPECTED_DIHEDRAL_ANGLE - cosDihedralAngle(planeA)
                    < COPLANAR_COS_ANGLE_THRESHOLD);
        }

        bool epsilonPerpendicular(Plane3_<number_t> planeA, number_t eps = PERPENDICULAR_SIN_ANGLE_THRESHOLD) const {
            return (std::abs(cosDihedralAngle(planeA)) < eps);
        }

        void interpolate(number_t alpha, Plane3_<number_t> planeA, Plane3_<number_t> planeB,
                cv::Point3_<number_t> pt) {
            this->x = alpha*planeA.x + (1 - alpha)*planeB.x;
            this->y = alpha*planeA.x + (1 - alpha)*planeB.x;
            this->z = alpha*planeA.x + (1 - alpha)*planeB.x;
            this->d = -(this->x*pt.x + this->y*pt.y + this->z*pt.z);
        }

        void convertHessianNormalForm() {
            number_t normScale = 1.0/std::sqrt(this->x*this->x +
                    this->y*this->y + this->z*this->z);
            scale(normScale);
        }

        friend std::ostream& operator<<(std::ostream& os, const Plane3_<number_t>& p) {
            os << "[" << p.x << ", " << p.y << ", " << p.z
                    << ", " << p.d << "]";
            return os;
        }

        cv::Point3_<number_t> uvToXYZ(const cv::Point_<number_t>& uv) {
            number_t threshold = 0.6; // > 1/sqrt(3)
            static cv::Point3_<number_t> uVec;
            static cv::Point3_<number_t> vVec;
            if (std::abs<number_t>(this->x) <= threshold) {
                number_t inverse = 1.0/std::sqrt(this->y*this->y + this->z*this->z);
                uVec = cv::Point3_<number_t>((number_t) 0, inverse*this->z, -inverse*this->y);
            } else if (std::abs<number_t>(this->y) <= threshold) {
                number_t inverse = 1.0/std::sqrt(this->x*this->x + this->z*this->z);
                uVec = cv::Point3_<number_t>(-inverse*this->z, (number_t) 0, inverse*this->x);
            } else {
                number_t inverse = 1.0/std::sqrt(this->x*this->x + this->y*this->y);
                uVec = cv::Point3_<number_t>(inverse*this->y, -inverse*this->x, (number_t) 0);
            }
            vVec = uVec.cross(*this);
            cv::Point3_<number_t> pt0(-d * this->x, -d * this->y, -d * this->z);
            pt0.x = pt0.x + uv.x * uVec.x + uv.y * vVec.x;
            pt0.y = pt0.y + uv.x * uVec.y + uv.y * vVec.y;
            pt0.z = pt0.z + uv.x * uVec.z + uv.y * vVec.z;
            return pt0;
        }

        cv::Point_<number_t> xyzToUV(const cv::Point3_<number_t>& p) {
            number_t threshold = 0.6; // > 1/sqrt(3)
            cv::Point_<number_t> uv;
            static cv::Point3_<number_t> uVec;
            static cv::Point3_<number_t> vVec;
            if (std::abs(this->x) <= threshold) {
                number_t inverse = 1.0 / std::sqrt(this->y*this->y + this->z*this->z);
                uVec = cv::Point3_<number_t>((number_t) 0.0, inverse*this->z, -inverse*this->y);
            } else if (std::abs(this->y) <= threshold) {
                number_t inverse = 1.0 / std::sqrt(this->x*this->x + this->z*this->z);
                uVec = cv::Point3_<number_t>(-inverse*this->z, (number_t) 0.0, inverse*this->x);
            } else {
                number_t inverse = 1.0 / std::sqrt(this->x*this->x + this->y*this->y);
                uVec = cv::Point3_<number_t>(inverse*this->y, -inverse*this->x, (number_t) 0.0);
            }
            vVec = uVec.cross(*this);
            cv::Point3_<number_t> pt0(-d*this->x, -d*this->y, -d*this->z);
            pt0 = p - pt0;
            uv.x = pt0.dot(uVec);
            uv.y = pt0.dot(vVec);
            return uv;
        }

        cv::Point3_<number_t> getClosestPointToOrigin() {
            return cv::Point3_<number_t>(-this->d*(*this));
        }

        std::string toString() {
            std::ostringstream stringStream;
            stringStream << "(" << this->x << ", " << this->y
                    << ", " << this->z << ", " << this->d << ")";
            return stringStream.str();
        }

        number_t d;
    };
    typedef Plane3_<float> Plane3f;
    typedef Plane3_<double> Plane3d;
    typedef Plane3_<int> Plane3i;
    
    template<typename number_t> class LabeledPlane3_ : public Plane3_<number_t> {
    public:

        typedef boost::shared_ptr<LabeledPlane3_<number_t>> Ptr;
        typedef boost::shared_ptr<const LabeledPlane3_<number_t>> ConstPtr;

        LabeledPlane3_() : label(0), Plane3_<number_t>() {

        };

        LabeledPlane3_(Plane3_<number_t> p, int _label) : label(_label), Plane3_<number_t>(p.x, p.y, p.z, p.d) {

        };

        LabeledPlane3_(number_t x, number_t y, number_t z, number_t d) : label(0),
        Plane3_<number_t>(x, y, z, d) {

        };

        LabeledPlane3_(number_t x, number_t y, number_t z, number_t d, int _label) : label(_label),
        Plane3_<number_t>(x, y, z, d) {

        };

        LabeledPlane3_<number_t> clone() {
            LabeledPlane3_<number_t> lp;
            lp.x = this->x;
            lp.y = this->y;
            lp.z = this->z;
            lp.d = this->d;
            lp.label = this->label;
            return lp;
        }

        number_t distance(number_t theta) {
            number_t my_theta = cv::fastAtan2(this->y, this->x); // degrees
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
        float error = 0;
        float error_squared = 0;
        float noise = 0;
        
        FitStatistics() {
        }
        
        FitStatistics(float _error, float _error_squared, float _noise, 
                size_t _inliers, size_t _outliers, size_t _invalid) : 
            error(_error), error_squared(_error_squared), noise(_noise), 
            Consensus::Consensus(_inliers, _outliers, _invalid) {}
        
    };
    
    template <typename number_t>
    class PlaneWithStats3_ : public LabeledPlane3_<number_t> {
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

