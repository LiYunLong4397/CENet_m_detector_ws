#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Core>
#include <Eigen/Dense>

typedef Eigen::Matrix3d M3D;
typedef Eigen::Vector3d V3D;
typedef Eigen::Matrix<double, 6, 1> V6D;

enum PointState {
    INVALID = -1,
    STATIC = 0,
    CASE1 = 1,
    CASE2 = 2,
    CASE3 = 3
};

enum SemanticLabel {
    UNLABELED = 0,
    CAR = 1,
    BICYCLE = 2,
    MOTORCYCLE = 3,
    TRUCK = 4,
    OTHER_VEHICLE = 5,
    PERSON = 6,
    BICYCLIST = 7,
    MOTORCYCLIST = 8,
    ROAD = 9,
    PARKING = 10,
    SIDEWALK = 11,
    OTHER_GROUND = 12,
    BUILDING = 13,
    FENCE = 14,
    VEGETATION = 15,
    TRUNK = 16,
    TERRAIN = 17,
    POLE = 18,
    TRAFFIC_SIGN = 19
};

struct point_soph
{
    V3D local;
    V3D glob;
    V3D vec;
    M3D rot;
    V3D pos;
    double time;
    int dyn;
    int occu_times;
    int is_occu_times;
    bool is_distort;
    int hor_ind;
    int ver_ind;
    int semantic_label;
    float semantic_confidence;
    
    typedef point_soph* Ptr;
    
    point_soph()
    {
        local. setZero();
        glob. setZero();
        vec. setZero();
        rot. setIdentity();
        pos. setZero();
        time = 0.0;
        dyn = STATIC;
        occu_times = 0;
        is_occu_times = 0;
        is_distort = false;
        hor_ind = 0;
        ver_ind = 0;
        semantic_label = UNLABELED;
        semantic_confidence = 0.0f;
    }
};

#endif