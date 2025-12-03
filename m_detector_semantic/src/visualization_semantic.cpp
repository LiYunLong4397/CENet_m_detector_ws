#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// 自定义点类型（匹配 M-Detector 输出）
struct PointXYZINormalCurvature
{
    PCL_ADD_POINT4D;
    float intensity;
    float normal_x;
    float normal_y;
    float normal_z;
    float curvature;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZINormalCurvature,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, normal_x, normal_x)
    (float, normal_y, normal_y)
    (float, normal_z, normal_z)
    (float, curvature, curvature)
)

ros::Publisher pub_semantic_colored;

// 语义类别颜色映射
void getSemanticColor(int label, uint8_t& r, uint8_t& g, uint8_t& b) {
    static const uint8_t colors[20][3] = {
        {0, 0, 0},       // 0: unlabeled - black
        {245, 150, 100}, // 1: car - red
        {245, 230, 100}, // 2: bicycle - yellow
        {150, 60, 30},   // 3: motorcycle - dark red
        {180, 30, 80},   // 4: truck - purple
        {255, 0, 0},     // 5: other-vehicle - bright red
        {30, 30, 255},   // 6: person - blue
        {200, 40, 255},  // 7: bicyclist - pink
        {90, 30, 150},   // 8: motorcyclist - dark purple
        {255, 0, 255},   // 9: road - magenta
        {255, 150, 255}, // 10: parking - light magenta
        {75, 0, 75},     // 11: sidewalk - dark magenta
        {75, 0, 175},    // 12: other-ground - dark blue
        {0, 200, 255},   // 13: building - cyan
        {50, 120, 255},  // 14: fence - light blue
        {0, 175, 0},     // 15: vegetation - green
        {0, 60, 135},    // 16: trunk - dark cyan
        {80, 240, 150},  // 17: terrain - light green
        {150, 240, 255}, // 18: pole - white-blue
        {0, 0, 255}      // 19: traffic-sign - pure blue
    };
    
    if (label >= 0 && label < 20) {
        r = colors[label][0];
        g = colors[label][1];
        b = colors[label][2];
    } else {
        r = g = b = 128; // gray for unknown
    }
}

void semanticCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    // 使用自定义点类型
    pcl::PointCloud<PointXYZINormalCurvature>::Ptr cloud(new pcl::PointCloud<PointXYZINormalCurvature>());
    pcl::fromROSMsg(*msg, *cloud);
    
    // 创建彩色点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    colored_cloud->reserve(cloud->size());
    
    for (const auto& point : cloud->points) {
        pcl::PointXYZRGB p;
        p.x = point.x;
        p.y = point.y;
        p.z = point.z;
        
        // curvature 字段存储语义标签
        int semantic_label = static_cast<int>(point.curvature);
        
        uint8_t r, g, b;
        getSemanticColor(semantic_label, r, g, b);
        
        p.r = r;
        p.g = g;
        p.b = b;
        
        colored_cloud->push_back(p);
    }
    
    // 发布彩色点云
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*colored_cloud, output);
    output.header = msg->header;
    pub_semantic_colored.publish(output);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "visualization_semantic");
    ros::NodeHandle nh;
    
    pub_semantic_colored = nh.advertise<sensor_msgs::PointCloud2>("/m_detector/semantic_colored", 10);
    ros::Subscriber sub = nh.subscribe("/m_detector/point_out", 10, semanticCallback);
    
    ROS_INFO("Semantic visualization node started");
    ROS_INFO("  Subscribing: /m_detector/point_out");
    ROS_INFO("  Publishing: /m_detector/semantic_colored");
    
    ros::spin();
    
    return 0;
}
