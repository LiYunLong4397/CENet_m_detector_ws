#ifndef DYN_OBJ_CLUSTER_H
#define DYN_OBJ_CLUSTER_H

#include <ros/ros.h> 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <std_msgs/Header.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <m-detector/types.h>
#include <m-detector/SemanticSegmentor.h>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

class DynObjCluster {
public:
    // Clustering parameters
    float cluster_distance = 0.5f;
    int min_cluster_size = 10;
    int min_cluster_size_dyn = 5;    // For dynamic semantic classes
    int min_cluster_size_static = 15; // For static semantic classes
    
    // Semantic-assisted clustering
    bool use_semantic = false;
    float semantic_cluster_distance_same = 0.3f;  // Same semantic class
    float semantic_cluster_distance_diff = 0.8f;  // Different semantic class
    
public:
    DynObjCluster();
    ~DynObjCluster();
    
    void init(ros::NodeHandle& nh);
    
    // Standard clustering
    void Clusterprocess(std::vector<int>& dyn_tag,
                       const PointCloudXYZI& dyn_points,
                       const PointCloudXYZI& raw_points_world,
                       const std_msgs::Header& header,
                       const M3D& rot_end,
                       const V3D& pos_end);
    
    // Semantic-assisted clustering
    void ClusterprocessWithSemantic(std::vector<int>& dyn_tag,
                                   const PointCloudXYZI& dyn_points,
                                   const std::vector<int>& semantic_labels,
                                   const std::vector<float>& confidences,
                                   const PointCloudXYZI& raw_points_world,
                                   const std_msgs::Header& header,
                                   const M3D& rot_end,
                                   const V3D& pos_end);
    
private:
    // DBSCAN clustering
    void dbscanClustering(const PointCloudXYZI& cloud,
                         std::vector<int>& labels,
                         float eps,
                         int min_pts);
    
    // Semantic-assisted DBSCAN
    void dbscanClusteringWithSemantic(const PointCloudXYZI& cloud,
                                     const std::vector<int>& semantic_labels,
                                     const std::vector<float>& confidences,
                                     std::vector<int>& labels,
                                     int min_pts);
    
    // Get semantic-based cluster distance
    float getSemanticClusterDistance(int label1, int label2, 
                                    float conf1, float conf2);
    
    // Filter small clusters
    void filterSmallClusters(std::vector<int>& labels,
                            const std::vector<int>& semantic_labels,
                            int min_size);
    
    // Region growing
    void expandCluster(const PointCloudXYZI& cloud,
                      int point_idx,
                      int cluster_id,
                      std::vector<int>& labels,
                      std::vector<bool>& visited,
                      pcl::KdTreeFLANN<PointType>& kdtree,
                      float eps,
                      int min_pts);
};

#endif // DYN_OBJ_CLUSTER_H