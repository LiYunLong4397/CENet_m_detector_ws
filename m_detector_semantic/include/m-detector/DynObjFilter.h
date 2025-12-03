#ifndef DYN_OBJ_FLT_H
#define DYN_OBJ_FLT_H

#include <omp.h>
#include <mutex>
#include <math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>

#include <m-detector/types.h>
#include <m-detector/SemanticSegmentor.h>
#include <m-detector/DynObjCluster.h>
#include <m-detector/parallel_q.h>

using namespace std;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// Depth map constants
#define MAX_1D 3600
#define MAX_1D_HALF 900
#define MAX_2D_N 3240000

// Depth map structure
struct DepthMap {
    typedef std::shared_ptr<DepthMap> Ptr;
    
    std::vector<std::vector<point_soph*>> depth_map;
    double start_time;
    double end_time;
    
    DepthMap() : depth_map(MAX_2D_N), start_time(0.0), end_time(0.0) {}
    
    void clear() {
        for (auto& pixel : depth_map) {
            pixel.clear();
        }
        start_time = 0.0;
        end_time = 0.0;
    }
};

class DynObjFilter {
public:
    // Point cloud storage
    std::deque<DepthMap::Ptr> depth_map_list;
    PARALLEL_Q<point_soph*> buffer;
    std::vector<point_soph*> point_soph_pointers;
    
    int points_num_perframe = 200000;
    int cur_point_soph_pointers = 0;
    int max_pointers_num = 0;
    int frame_num_for_rec = 0;
    
    std::deque<PointCloudXYZI::Ptr> pcl_his_list;
    PointCloudXYZI::Ptr laserCloudSteadObj;
    PointCloudXYZI::Ptr laserCloudSteadObj_hist;
    PointCloudXYZI::Ptr laserCloudDynObj;
    PointCloudXYZI::Ptr laserCloudDynObj_world;
    PointCloudXYZI::Ptr laserCloudDynObj_clus;
    PointCloudXYZI::Ptr laserCloudSteadObj_clus;
    
    std::deque<PointCloudXYZI::Ptr> laserCloudSteadObj_accu;
    int laserCloudSteadObj_accu_times = 0;
    int laserCloudSteadObj_accu_limit = 10;
    
    // Clustering
    DynObjCluster Cluster;
    bool cluster_coupled = true;
    bool cluster_future = false;
    float voxel_filter_size = 0.5f;
    
    // Parameters
    double frame_dur = 0.1;
    double buffer_delay = 0.1;
    double depth_map_dur = 0.2f;
    int buffer_size = 300000;
    int max_depth_map_num = 5;
    int hor_num = MAX_1D;
    int ver_num = MAX_1D_HALF;
    float hor_resolution_max = 0.02f;
    float ver_resolution_max = 0.02f;
    
    // Thresholds
    int occu_time_th = 3;
    int is_occu_time_th = 3;
    int map_index = 0;
    int case1_num = 0, case2_num = 0, case3_num = 0;
    
    // Case 1 parameters
    float occluded_map_thr1 = 3.0f;
    
    // Case 2 parameters
    int occluded_times_thr2 = 3;
    float occ_depth_thr2 = 0.15f;
    float occ_hor_thr2 = 0.02f;
    float occ_ver_thr2 = 0.01f;
    float depth_cons_depth_thr2 = 0.15f;
    float depth_cons_hor_thr2 = 0.02f;
    float depth_cons_ver_thr2 = 0.01f;
    int depth_cons_hor_num2 = 0;
    int depth_cons_ver_num2 = 0;
    float k_depth2 = 0.005f;
    
    // Case 3 parameters
    int occluding_times_thr3 = 3;
    float occ_depth_thr3 = 0.15f;
    float occ_hor_thr3 = 0.02f;
    float occ_ver_thr3 = 0.01f;
    float map_cons_depth_thr3 = 0.15f;
    float map_cons_hor_thr3 = 0.02f;
    float map_cons_ver_thr3 = 0.01f;
    float depth_cons_depth_thr3 = 0.15f;
    float depth_cons_depth_max_thr3 = 0.15f;
    float depth_cons_hor_thr3 = 0.02f;
    float depth_cons_ver_thr3 = 0.01f;
    int depth_cons_hor_num3 = 0;
    int depth_cons_ver_num3 = 0;
    float k_depth3 = 0.005f;
    float k_depth_max_thr3 = 0.02f;
    float d_depth_max_thr3 = 1.0f;
    float cutoff_value = 0.1f;
    float v_min_thr3 = 1.0f;
    float enlarge_distort = 1.5f;
    
    // FOV and sensor parameters
    float fov_up = 2.0f;
    float fov_down = -23.0f;
    float fov_cut = -20.0f;
    float fov_left = 180.0f;
    float fov_right = -180.0f;
    float blind_dis = 0.3f;
    int pixel_fov_up, pixel_fov_down, pixel_fov_cut, pixel_fov_left, pixel_fov_right;
    int max_pixel_points = 50;
    bool stop_object_detect = false;
    point_soph::Ptr last_point_pointer = nullptr;
    string frame_id = "camera_init";
    
    // Timing
    double time_total = 0.0;
    double time_total_avr = 0.0;
    
    // Dataset and vehicle parameters
    int dataset = 0;
    float self_x_f = 2.5f;
    float self_x_b = -1.5f;
    float self_y_l = 1.6f;
    float self_y_r = -1.6f;
    
    bool dyn_filter_en = true;
    bool debug_en = false;
    
    // Semantic segmentation
    std::shared_ptr<SemanticSegmentor> semantic_segmentor;
    bool use_semantic = false;
    std::vector<int> current_semantic_labels;
    std::vector<float> current_semantic_confidences;
    
    // Semantic-assisted parameters
    float semantic_dyn_weight = 0.3f;
    float occluded_map_thr1_dyn = 2.0f;
    float occluded_map_thr1_static = 5.0f;
    float semantic_conf_threshold = 0.5f;
    
    // Mutexes
    mutex mtx_case2, mtx_case3;
    
    // Output files
    bool is_set_path = false;
    string out_file;
    string out_file_origin;
    ofstream time_out;
    
public:
    DynObjFilter();
    DynObjFilter(float windows_dur, float hor_resolution, float ver_resolution);
    ~DynObjFilter();
    
    void init(ros::NodeHandle& nh);
    bool initSemanticSegmentor(const std::string& model_path, const std::string& config_path);
    
    void filter(PointCloudXYZI::Ptr feats_undistort, const M3D& rot_end, 
                const V3D& pos_end, const double& scan_end_time);
    
    void publish_dyn(const ros::Publisher& pub_point_out, 
                     const ros::Publisher& pub_frame_out, 
                     const ros::Publisher& pub_steady_points, 
                     const double& scan_end_time);
    
    void set_path(string file_path, string file_path_origin);
    
private:
    // Core detection functions
    bool Case1(point_soph& p);
    bool Case2(point_soph& p);
    bool Case3(point_soph& p);
    
    // Semantic-assisted detection functions
    bool Case1WithSemantic(point_soph& p, int semantic_label, float confidence);
    bool Case2WithSemantic(point_soph& p, int semantic_label, float confidence);
    bool Case3WithSemantic(point_soph& p, int semantic_label, float confidence);
    
    // Helper functions
    bool Case2IsOccluded(const point_soph& p, const point_soph& p_occ);
    bool Case2DepthConsistencyCheck(const point_soph& p, const DepthMap& map_info);
    bool Case3IsOccluding(const point_soph& p, const point_soph& p_occ);
    bool Case3DepthConsistencyCheck(const point_soph& p, const DepthMap& map_info);
    
    float computeSemanticDynamicScore(int semantic_label, float confidence, float original_score);
    float getAdaptiveThreshold(int semantic_label, float confidence, float base_threshold);
    
    void buildDepthMap(const std::vector<point_soph*>& points, DepthMap::Ptr& depth_map);
    void updateDepthMapList(double current_time);
};

#endif // DYN_OBJ_FLT_H