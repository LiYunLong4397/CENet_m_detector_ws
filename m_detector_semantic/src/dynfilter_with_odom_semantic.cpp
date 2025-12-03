#include <ros/ros.h>
#include <omp.h>
#include <mutex>
#include <deque>
#include <Eigen/Core>
#include <m-detector/DynObjFilter.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using namespace std;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// Global variables
shared_ptr<DynObjFilter> DynObjFilt(new DynObjFilter());
M3D cur_rot = Eigen::Matrix3d::Identity();
V3D cur_pos = Eigen::Vector3d::Zero();

string points_topic, odom_topic;
string out_folder, out_folder_origin;
double lidar_end_time = 0.0;
int cur_frame = 0;

deque<M3D> buffer_rots;
deque<V3D> buffer_poss;
deque<double> buffer_times;
deque<boost::shared_ptr<PointCloudXYZI>> buffer_pcs;

ros::Publisher pub_pcl_dyn;
ros::Publisher pub_pcl_dyn_extend;
ros::Publisher pub_pcl_std;

mutex mtx_buffer;

void PointsCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
    pcl::fromROSMsg(*msg, *cloud);
    
    double timestamp = msg->header.stamp.toSec();
    
    lock_guard<mutex> lock(mtx_buffer);
    buffer_pcs.push_back(cloud);
    buffer_times.push_back(timestamp);
}

void OdomCallback(const nav_msgs::OdometryConstPtr& msg) {
    M3D rot;
    V3D pos;
    
    tf::Quaternion q(msg->pose.pose.orientation. x,
                     msg->pose. pose.orientation.y,
                     msg->pose.pose.orientation.z,
                     msg->pose.pose.orientation.w);
    tf::Matrix3x3 mat(q);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rot(i, j) = mat[i][j];
        }
    }
    
    pos << msg->pose.pose.position. x,
           msg->pose.pose.position. y,
           msg->pose. pose.position.z;
    
    double timestamp = msg->header.stamp. toSec();
    
    lock_guard<mutex> lock(mtx_buffer);
    buffer_rots.push_back(rot);
    buffer_poss.push_back(pos);
    
    // Match with point cloud
    while (!buffer_pcs.empty() && ! buffer_times.empty() && 
           ! buffer_rots.empty() && !buffer_poss.empty()) {
        
        double pc_time = buffer_times.front();
        double odom_time = timestamp;
        
        if (fabs(pc_time - odom_time) < 0.01) {
            // Process
            PointCloudXYZI::Ptr cloud = buffer_pcs.front();
            M3D rot_cur = buffer_rots.front();
            V3D pos_cur = buffer_poss.front();
            
            DynObjFilt->filter(cloud, rot_cur, pos_cur, pc_time);
            DynObjFilt->publish_dyn(pub_pcl_dyn, pub_pcl_dyn_extend, 
                                   pub_pcl_std, pc_time);
            
            buffer_pcs.pop_front();
            buffer_times.pop_front();
            buffer_rots.pop_front();
            buffer_poss.pop_front();
            
            cur_frame++;
            
        } else if (pc_time < odom_time - 0.01) {
            buffer_pcs.pop_front();
            buffer_times.pop_front();
        } else {
            break;
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "m_detector_semantic");
    ros::NodeHandle nh;
    
    // Load parameters
    nh.param<string>("dyn_obj/points_topic", points_topic, "/velodyne_points");
    nh.param<string>("dyn_obj/odom_topic", odom_topic, "/Odometry");
    nh. param<string>("dyn_obj/out_file", out_folder, "");
    nh.param<string>("dyn_obj/out_file_origin", out_folder_origin, "");
    
    bool use_semantic;
    string semantic_model_path, semantic_config_path;
    nh.param<bool>("dyn_obj/use_semantic", use_semantic, false);
    nh.param<string>("dyn_obj/semantic_model_path", semantic_model_path, "");
    nh.param<string>("dyn_obj/semantic_config_path", semantic_config_path, "");
    
    // Initialize filter
    DynObjFilt->init(nh);
    
    // Initialize semantic segmentation
    if (use_semantic && ! semantic_model_path.empty() && !semantic_config_path. empty()) {
        if (! DynObjFilt->initSemanticSegmentor(semantic_model_path, semantic_config_path)) {
            ROS_WARN("Running without semantic segmentation");
        }
    } else {
        ROS_INFO("Semantic segmentation disabled");
    }
    
    // Set output paths
    if (! out_folder.empty()) {
        DynObjFilt->set_path(out_folder, out_folder_origin);
    }
    
    // Publishers
    pub_pcl_dyn = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/point_out", 100);
    pub_pcl_dyn_extend = nh. advertise<sensor_msgs::PointCloud2>("/m_detector/frame_out", 100);
    pub_pcl_std = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/std_points", 100);
    
    // Subscribers
    ros::Subscriber sub_pcl = nh.subscribe(points_topic, 200, PointsCallback);
    ros::Subscriber sub_odom = nh.subscribe(odom_topic, 200, OdomCallback);
    
    ROS_INFO("M-Detector with Semantic Segmentation started");
    ROS_INFO("Point cloud topic: %s", points_topic. c_str());
    ROS_INFO("Odometry topic: %s", odom_topic. c_str());
    
    ros::spin();
    
    return 0;
}
