#include <m-detector/DynObjCluster.h>
#include <ros/ros.h>

DynObjCluster::DynObjCluster() {
}

DynObjCluster::~DynObjCluster() {
}

void DynObjCluster::init(ros::NodeHandle& nh) {
    nh.param<float>("dyn_obj/cluster_distance", cluster_distance, 0.5f);
    nh.param<int>("dyn_obj/min_cluster_size", min_cluster_size, 10);
    nh.param<int>("dyn_obj/min_cluster_size_dyn", min_cluster_size_dyn, 5);
    nh.param<int>("dyn_obj/min_cluster_size_static", min_cluster_size_static, 15);
    nh.param<bool>("dyn_obj/use_semantic", use_semantic, false);
    nh.param<float>("dyn_obj/semantic_cluster_distance_same", semantic_cluster_distance_same, 0.3f);
    nh.param<float>("dyn_obj/semantic_cluster_distance_diff", semantic_cluster_distance_diff, 0.8f);
    
    ROS_INFO("DynObjCluster initialized. Semantic clustering: %s", use_semantic ?  "ON" : "OFF");
}

float DynObjCluster::getSemanticClusterDistance(int label1, int label2, 
                                               float conf1, float conf2) {
    // If low confidence, use default distance
    if (conf1 < 0.5f || conf2 < 0.5f) {
        return cluster_distance;
    }
    
    // Same semantic class - tighter clustering
    if (label1 == label2 && SemanticSegmentor::isDynamicClass(label1)) {
        return semantic_cluster_distance_same;
    }
    
    // Both dynamic but different classes
    if (SemanticSegmentor::isDynamicClass(label1) && 
        SemanticSegmentor::isDynamicClass(label2)) {
        return (semantic_cluster_distance_same + cluster_distance) / 2.0f;
    }
    
    // Different semantic types - looser clustering
    return semantic_cluster_distance_diff;
}

void DynObjCluster::expandCluster(const PointCloudXYZI& cloud,
                                 int point_idx,
                                 int cluster_id,
                                 std::vector<int>& labels,
                                 std::vector<bool>& visited,
                                 pcl::KdTreeFLANN<PointType>& kdtree,
                                 float eps,
                                 int min_pts) {
    std::vector<int> seeds;
    seeds.push_back(point_idx);
    labels[point_idx] = cluster_id;
    
    for (size_t i = 0; i < seeds.size(); i++) {
        int current_point = seeds[i];
        
        std::vector<int> neighbors;
        std::vector<float> distances;
        kdtree.radiusSearch(cloud.points[current_point], eps, neighbors, distances);
        
        if (neighbors.size() >= static_cast<size_t>(min_pts)) {
            for (size_t j = 0; j < neighbors.size(); j++) {
                int neighbor_idx = neighbors[j];
                
                if (! visited[neighbor_idx]) {
                    visited[neighbor_idx] = true;
                    seeds.push_back(neighbor_idx);
                }
                
                if (labels[neighbor_idx] == 0) {
                    labels[neighbor_idx] = cluster_id;
                }
            }
        }
    }
}

void DynObjCluster::dbscanClustering(const PointCloudXYZI& cloud,
                                    std::vector<int>& labels,
                                    float eps,
                                    int min_pts) {
    int n_points = cloud.points.size();
    labels.assign(n_points, 0);
    std::vector<bool> visited(n_points, false);
    
    pcl::KdTreeFLANN<PointType> kdtree;
    PointCloudXYZI::Ptr cloud_ptr(new PointCloudXYZI(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    int cluster_id = 1;
    
    for (int i = 0; i < n_points; i++) {
        if (visited[i]) continue;
        
        visited[i] = true;
        
        std::vector<int> neighbors;
        std::vector<float> distances;
        kdtree.radiusSearch(cloud.points[i], eps, neighbors, distances);
        
        if (neighbors.size() >= static_cast<size_t>(min_pts)) {
            expandCluster(cloud, i, cluster_id, labels, visited, kdtree, eps, min_pts);
            cluster_id++;
        }
    }
}

void DynObjCluster::dbscanClusteringWithSemantic(const PointCloudXYZI& cloud,
                                                const std::vector<int>& semantic_labels,
                                                const std::vector<float>& confidences,
                                                std::vector<int>& labels,
                                                int min_pts) {
    int n_points = cloud.points. size();
    labels.assign(n_points, 0);
    std::vector<bool> visited(n_points, false);
    
    pcl::KdTreeFLANN<PointType> kdtree;
    PointCloudXYZI::Ptr cloud_ptr(new PointCloudXYZI(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    int cluster_id = 1;
    
    for (int i = 0; i < n_points; i++) {
        if (visited[i]) continue;
        
        visited[i] = true;
        
        // Adaptive search radius based on semantic label
        float search_radius = getSemanticClusterDistance(
            semantic_labels[i], semantic_labels[i], 
            confidences[i], confidences[i]);
        
        std::vector<int> neighbors;
        std::vector<float> distances;
        kdtree. radiusSearch(cloud.points[i], search_radius * 2.0f, neighbors, distances);
        
        // Filter neighbors by semantic-aware distance
        std::vector<int> valid_neighbors;
        for (size_t j = 0; j < neighbors.size(); j++) {
            int neighbor_idx = neighbors[j];
            float sem_dist = getSemanticClusterDistance(
                semantic_labels[i], semantic_labels[neighbor_idx],
                confidences[i], confidences[neighbor_idx]);
            
            if (distances[j] <= sem_dist * sem_dist) {
                valid_neighbors.push_back(neighbor_idx);
            }
        }
        
        if (valid_neighbors.size() >= static_cast<size_t>(min_pts)) {
            // Start new cluster
            std::vector<int> seeds;
            seeds.push_back(i);
            labels[i] = cluster_id;
            
            for (size_t k = 0; k < seeds.size(); k++) {
                int current_point = seeds[k];
                
                float current_radius = getSemanticClusterDistance(
                    semantic_labels[current_point], semantic_labels[current_point],
                    confidences[current_point], confidences[current_point]);
                
                std::vector<int> curr_neighbors;
                std::vector<float> curr_distances;
                kdtree.radiusSearch(cloud.points[current_point], current_radius * 2.0f, 
                                  curr_neighbors, curr_distances);
                
                for (size_t m = 0; m < curr_neighbors.size(); m++) {
                    int neighbor_idx = curr_neighbors[m];
                    
                    float sem_dist = getSemanticClusterDistance(
                        semantic_labels[current_point], semantic_labels[neighbor_idx],
                        confidences[current_point], confidences[neighbor_idx]);
                    
                    if (curr_distances[m] > sem_dist * sem_dist) continue;
                    
                    if (! visited[neighbor_idx]) {
                        visited[neighbor_idx] = true;
                        seeds. push_back(neighbor_idx);
                    }
                    
                    if (labels[neighbor_idx] == 0) {
                        labels[neighbor_idx] = cluster_id;
                    }
                }
            }
            
            cluster_id++;
        }
    }
}

void DynObjCluster::filterSmallClusters(std::vector<int>& labels,
                                       const std::vector<int>& semantic_labels,
                                       int min_size) {
    std::map<int, int> cluster_sizes;
    std::map<int, int> cluster_semantic_types;
    
    // Count cluster sizes and determine dominant semantic type
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] > 0) {
            cluster_sizes[labels[i]]++;
            if (cluster_semantic_types.find(labels[i]) == cluster_semantic_types. end()) {
                cluster_semantic_types[labels[i]] = semantic_labels[i];
            }
        }
    }
    
    // Filter clusters
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] > 0) {
            int cluster_id = labels[i];
            int semantic_type = cluster_semantic_types[cluster_id];
            
            // Dynamic objects (e.g., pedestrians) allow smaller clusters
            int adaptive_min_size = min_size;
            if (use_semantic) {
                if (SemanticSegmentor::isDynamicClass(semantic_type)) {
                    adaptive_min_size = min_cluster_size_dyn;
                } else {
                    adaptive_min_size = min_cluster_size_static;
                }
            }
            
            if (cluster_sizes[cluster_id] < adaptive_min_size) {
                labels[i] = 0; // Mark as non-clustered
            }
        }
    }
}

void DynObjCluster::Clusterprocess(std::vector<int>& dyn_tag,
                                  const PointCloudXYZI& dyn_points,
                                  const PointCloudXYZI& raw_points_world,
                                  const std_msgs::Header& header,
                                  const M3D& rot_end,
                                  const V3D& pos_end) {
    if (dyn_points.empty()) {
        dyn_tag.clear();
        return;
    }
    
    dbscanClustering(dyn_points, dyn_tag, cluster_distance, min_cluster_size);
    
    // Filter small clusters
    std::vector<int> dummy_semantic(dyn_points.size(), UNLABELED);
    filterSmallClusters(dyn_tag, dummy_semantic, min_cluster_size);
}

void DynObjCluster::ClusterprocessWithSemantic(std::vector<int>& dyn_tag,
                                              const PointCloudXYZI& dyn_points,
                                              const std::vector<int>& semantic_labels,
                                              const std::vector<float>& confidences,
                                              const PointCloudXYZI& raw_points_world,
                                              const std_msgs::Header& header,
                                              const M3D& rot_end,
                                              const V3D& pos_end) {
    if (dyn_points.empty()) {
        dyn_tag. clear();
        return;
    }
    
    if (! use_semantic || semantic_labels.size() != dyn_points.size()) {
        // Fallback to standard clustering
        Clusterprocess(dyn_tag, dyn_points, raw_points_world, header, rot_end, pos_end);
        return;
    }
    
    dbscanClusteringWithSemantic(dyn_points, semantic_labels, confidences, 
                                dyn_tag, min_cluster_size);
    
    filterSmallClusters(dyn_tag, semantic_labels, min_cluster_size);
}
