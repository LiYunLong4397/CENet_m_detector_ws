#ifndef SEMANTIC_SEGMENTOR_H
#define SEMANTIC_SEGMENTOR_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <m-detector/types.h>

class SemanticSegmentor {
public:
    SemanticSegmentor();
    ~SemanticSegmentor();
    
    bool init(const std::string& model_path, const std::string& config_path);
    
    bool segment(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
                 std::vector<int>& semantic_labels,
                 std::vector<float>& confidences);
    
    static bool isDynamicClass(int label);
    static bool isStaticClass(int label);
    static float getDynamicPrior(int label);
    static std::string getClassName(int label);
    static void getClassColor(int label, uint8_t& r, uint8_t& g, uint8_t& b);
    
    bool isInitialized() const { return initialized; }
    
private:
    PyObject* pModule;
    PyObject* pInferenceFunc;
    bool initialized;
    std::mutex python_mutex;
    
    PyObject* convertPCLToNumpy(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    bool initPython();
    void cleanupPython();
};

#endif