#include <m-detector/SemanticSegmentor.h>
#include <ros/ros.h>
#include <iostream>

SemanticSegmentor::SemanticSegmentor() 
    : pModule(nullptr), pInferenceFunc(nullptr), initialized(false) {
}

SemanticSegmentor::~SemanticSegmentor() {
    cleanupPython();
}

bool SemanticSegmentor::initPython() {
    if (! Py_IsInitialized()) {
        Py_Initialize();
        if (!Py_IsInitialized()) {
            ROS_ERROR("Failed to initialize Python interpreter");
            return false;
        }
        
        // Initialize NumPy
        import_array1(false);
    }
    return true;
}

void SemanticSegmentor::cleanupPython() {
    std::lock_guard<std::mutex> lock(python_mutex);
    
    if (pInferenceFunc) {
        Py_DECREF(pInferenceFunc);
        pInferenceFunc = nullptr;
    }
    if (pModule) {
        Py_DECREF(pModule);
        pModule = nullptr;
    }
    
    initialized = false;
}

bool SemanticSegmentor::init(const std::string& model_path, const std::string& config_path) {
    std::lock_guard<std::mutex> lock(python_mutex);
    
    if (!initPython()) {
        return false;
    }
    
    // Add script path to Python system path
    std::string script_dir = model_path.substr(0, model_path.find_last_of("/"));
    std::string add_path_cmd = "import sys; sys.path.insert(0, '" + script_dir + "')";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(add_path_cmd.c_str());
    
    ROS_INFO("Loading CENet module from: %s", script_dir.c_str());
    
    // Import the cenet_inference module
    PyObject* pName = PyUnicode_DecodeFSDefault("cenet_inference");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    
    if (!pModule) {
        PyErr_Print();
        ROS_ERROR("Failed to load CENet Python module");
        return false;
    }
    
    // Get init_model function
    PyObject* pInitFunc = PyObject_GetAttrString(pModule, "init_model");
    if (!pInitFunc || !PyCallable_Check(pInitFunc)) {
        if (PyErr_Occurred()) PyErr_Print();
        ROS_ERROR("Cannot find function 'init_model'");
        Py_XDECREF(pInitFunc);
        return false;
    }
    
    // Call init_model
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(model_path.c_str()));
    PyTuple_SetItem(pArgs, 1, PyUnicode_FromString(config_path.c_str()));
    
    PyObject* pInitResult = PyObject_CallObject(pInitFunc, pArgs);
    Py_DECREF(pArgs);
    Py_DECREF(pInitFunc);
    
    if (!pInitResult) {
        PyErr_Print();
        ROS_ERROR("Failed to initialize CENet model");
        return false;
    }
    Py_DECREF(pInitResult);
    
    // Get inference function
    pInferenceFunc = PyObject_GetAttrString(pModule, "inference");
    if (!pInferenceFunc || !PyCallable_Check(pInferenceFunc)) {
        if (PyErr_Occurred()) PyErr_Print();
        ROS_ERROR("Cannot find function 'inference'");
        return false;
    }
    
    initialized = true;
    ROS_INFO("CENet semantic segmentor initialized successfully");
    return true;
}

PyObject* SemanticSegmentor::convertPCLToNumpy(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    int n_points = cloud->points.size();
    npy_intp dims[2] = {n_points, 4};
    
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    float* data = (float*)PyArray_DATA((PyArrayObject*)array);
    
    for (int i = 0; i < n_points; ++i) {
        data[i * 4 + 0] = cloud->points[i].x;
        data[i * 4 + 1] = cloud->points[i].y;
        data[i * 4 + 2] = cloud->points[i].z;
        data[i * 4 + 3] = cloud->points[i]. intensity;
    }
    
    return array;
}

bool SemanticSegmentor::segment(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                               std::vector<int>& semantic_labels,
                               std::vector<float>& confidences) {
    std::lock_guard<std::mutex> lock(python_mutex);
    
    if (!initialized) {
        ROS_WARN_THROTTLE(5.0, "SemanticSegmentor not initialized");
        return false;
    }
    
    if (cloud->points.empty()) {
        ROS_WARN("Empty point cloud provided to semantic segmentor");
        return false;
    }
    
    // Convert PCL to numpy
    PyObject* pArray = convertPCLToNumpy(cloud);
    if (!pArray) {
        ROS_ERROR("Failed to convert point cloud to numpy array");
        return false;
    }
    
    // Call inference function
    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pArray);
    
    PyObject* pResult = PyObject_CallObject(pInferenceFunc, pArgs);
    Py_DECREF(pArgs);
    
    if (!pResult) {
        PyErr_Print();
        ROS_ERROR("CENet inference failed");
        return false;
    }
    
    // Parse result (should be a tuple of (labels, confidences))
    if (!PyTuple_Check(pResult)) {
        ROS_ERROR("Unexpected inference result format");
        Py_DECREF(pResult);
        return false;
    }
    
    PyArrayObject* labels_array = (PyArrayObject*)PyTuple_GetItem(pResult, 0);
    PyArrayObject* conf_array = (PyArrayObject*)PyTuple_GetItem(pResult, 1);
    
    int n_points = PyArray_DIM(labels_array, 0);
    int* labels_data = (int*)PyArray_DATA(labels_array);
    float* conf_data = (float*)PyArray_DATA(conf_array);
    
    semantic_labels.resize(n_points);
    confidences.resize(n_points);
    
    for (int i = 0; i < n_points; ++i) {
        semantic_labels[i] = labels_data[i];
        confidences[i] = conf_data[i];
    }
    
    Py_DECREF(pResult);
    return true;
}

bool SemanticSegmentor::isDynamicClass(int label) {
    return (label >= CAR && label <= MOTORCYCLIST);
}

bool SemanticSegmentor::isStaticClass(int label) {
    return (label >= ROAD && label <= TRAFFIC_SIGN);
}

float SemanticSegmentor::getDynamicPrior(int label) {
    // Dynamic object classes: higher dynamic tendency
    switch(label) {
        case PERSON:
        case BICYCLIST:
        case MOTORCYCLIST:
            return 0.9f;
        case CAR:
        case TRUCK:
        case OTHER_VEHICLE:
            return 0.8f;
        case BICYCLE:
        case MOTORCYCLE:
            return 0.7f;
        // Static object classes: lower dynamic tendency
        case ROAD:
        case PARKING:
        case SIDEWALK:
            return 0.05f;
        case BUILDING:
        case FENCE:
            return 0.1f;
        case VEGETATION:
        case TRUNK:
        case TERRAIN:
            return 0.15f;
        case POLE:
        case TRAFFIC_SIGN:
            return 0.2f;
        default:
            return 0.5f; // Unknown/unlabeled
    }
}

std::string SemanticSegmentor::getClassName(int label) {
    static const std::string class_names[] = {
        "unlabeled", "car", "bicycle", "motorcycle", "truck", "other-vehicle",
        "person", "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
        "other-ground", "building", "fence", "vegetation", "trunk", "terrain",
        "pole", "traffic-sign"
    };
    
    if (label >= 0 && label < 20) {
        return class_names[label];
    }
    return "unknown";
}

void SemanticSegmentor::getClassColor(int label, uint8_t& r, uint8_t& g, uint8_t& b) {
    // SemanticKITTI color scheme
    static const uint8_t colors[][3] = {
        {0, 0, 0},       // unlabeled - black
        {245, 150, 100}, // car - coral
        {245, 230, 100}, // bicycle - yellow
        {150, 60, 30},   // motorcycle - brown
        {180, 30, 80},   // truck - purple
        {255, 0, 0},     // other-vehicle - red
        {30, 30, 255},   // person - blue
        {200, 40, 255},  // bicyclist - pink
        {90, 30, 150},   // motorcyclist - violet
        {255, 0, 255},   // road - magenta
        {255, 150, 255}, // parking - light magenta
        {75, 0, 75},     // sidewalk - dark purple
        {75, 0, 175},    // other-ground - purple
        {0, 200, 255},   // building - cyan
        {50, 120, 255},  // fence - light blue
        {0, 175, 0},     // vegetation - green
        {0, 60, 135},    // trunk - dark blue
        {80, 240, 150},  // terrain - light green
        {150, 240, 255}, // pole - light cyan
        {0, 0, 255}      // traffic-sign - blue
    };
    
    if (label >= 0 && label < 20) {
        r = colors[label][0];
        g = colors[label][1];
        b = colors[label][2];
    } else {
        r = g = b = 128; // gray for unknown
    }
}
