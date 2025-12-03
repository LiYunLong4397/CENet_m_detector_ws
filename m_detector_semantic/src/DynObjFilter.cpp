#include <m-detector/DynObjFilter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

DynObjFilter::DynObjFilter() 
    : DynObjFilter(0.2f, 0.02f, 0.02f) {
}

DynObjFilter::DynObjFilter(float windows_dur, float hor_resolution, float ver_resolution)
    : depth_map_dur(windows_dur), 
      hor_resolution_max(hor_resolution), 
      ver_resolution_max(ver_resolution) {
    
    laserCloudSteadObj. reset(new PointCloudXYZI());
    laserCloudSteadObj_hist.reset(new PointCloudXYZI());
    laserCloudDynObj.reset(new PointCloudXYZI());
    laserCloudDynObj_world.reset(new PointCloudXYZI());
    laserCloudDynObj_clus. reset(new PointCloudXYZI());
    laserCloudSteadObj_clus.reset(new PointCloudXYZI());
    
    semantic_segmentor = std::make_shared<SemanticSegmentor>();
}

DynObjFilter::~DynObjFilter() {
    for (auto ptr : point_soph_pointers) {
        if (ptr) delete ptr;
    }
    point_soph_pointers. clear();
}

void DynObjFilter::init(ros::NodeHandle& nh) {
    nh.param<int>("dyn_obj/dataset", dataset, 0);
    nh.param<bool>("dyn_obj/dyn_filter_en", dyn_filter_en, true);
    nh.param<bool>("dyn_obj/debug_en", debug_en, false);
    nh.param<bool>("dyn_obj/cluster_coupled", cluster_coupled, true);
    nh.param<bool>("dyn_obj/cluster_future", cluster_future, false);
    
    nh.param<float>("dyn_obj/fov_up", fov_up, 2.0f);
    nh.param<float>("dyn_obj/fov_down", fov_down, -23.0f);
    nh.param<float>("dyn_obj/fov_cut", fov_cut, -20.0f);
    nh.param<float>("dyn_obj/blind_dis", blind_dis, 0.3f);
    
    nh.param<double>("dyn_obj/frame_dur", frame_dur, 0.1);
    nh.param<double>("dyn_obj/depth_map_dur", depth_map_dur, 0.2);
    nh.param<int>("dyn_obj/max_depth_map_num", max_depth_map_num, 5);
    
    nh.param<float>("dyn_obj/hor_resolution_max", hor_resolution_max, 0.02f);
    nh.param<float>("dyn_obj/ver_resolution_max", ver_resolution_max, 0.02f);
    
    nh.param<float>("dyn_obj/occluded_map_thr1", occluded_map_thr1, 3.0f);
    nh.param<int>("dyn_obj/occluded_times_thr2", occluded_times_thr2, 3);
    nh.param<int>("dyn_obj/occluding_times_thr3", occluding_times_thr3, 3);
    
    nh.param<bool>("dyn_obj/use_semantic", use_semantic, false);
    nh.param<float>("dyn_obj/semantic_dyn_weight", semantic_dyn_weight, 0.3f);
    nh.param<float>("dyn_obj/occluded_map_thr1_dyn", occluded_map_thr1_dyn, 2.0f);
    nh.param<float>("dyn_obj/occluded_map_thr1_static", occluded_map_thr1_static, 5.0f);
    nh.param<float>("dyn_obj/semantic_conf_threshold", semantic_conf_threshold, 0.5f);
    
    nh.param<float>("dyn_obj/self_x_f", self_x_f, 2.5f);
    nh.param<float>("dyn_obj/self_x_b", self_x_b, -1.5f);
    nh.param<float>("dyn_obj/self_y_l", self_y_l, 1.6f);
    nh.param<float>("dyn_obj/self_y_r", self_y_r, -1.6f);
    
    nh.param<std::string>("dyn_obj/frame_id", frame_id, "camera_init");
    
    Cluster. init(nh);
    Cluster. use_semantic = use_semantic;
    
    max_pointers_num = points_num_perframe * (max_depth_map_num + 2);
    point_soph_pointers.resize(max_pointers_num);
    for (int i = 0; i < max_pointers_num; i++) {
        point_soph_pointers[i] = new point_soph();
    }
    
    ROS_INFO("DynObjFilter initialized.  Semantic: %s", use_semantic ? "ENABLED" : "DISABLED");
}

bool DynObjFilter::initSemanticSegmentor(const std::string& model_path, 
                                        const std::string& config_path) {
    if (!use_semantic) {
        ROS_WARN("Semantic segmentation is disabled in config");
        return false;
    }
    
    bool success = semantic_segmentor->init(model_path, config_path);
    if (success) {
        ROS_INFO("Semantic segmentation enabled with CENet");
        use_semantic = true;
    } else {
        ROS_WARN("Failed to initialize semantic segmentation");
        use_semantic = false;
    }
    return success;
}

float DynObjFilter::computeSemanticDynamicScore(int semantic_label, float confidence, 
                                               float original_score) {
    if (confidence < semantic_conf_threshold) {
        return original_score;
    }
    
    float semantic_prior = SemanticSegmentor::getDynamicPrior(semantic_label);
    float weight = semantic_dyn_weight * confidence;
    
    return (1.0f - weight) * original_score + weight * semantic_prior;
}

float DynObjFilter::getAdaptiveThreshold(int semantic_label, float confidence, 
                                        float base_threshold) {
    if (confidence < semantic_conf_threshold) {
        return base_threshold;
    }
    
    if (SemanticSegmentor::isDynamicClass(semantic_label)) {
        return base_threshold * (occluded_map_thr1_dyn / occluded_map_thr1);
    } else if (SemanticSegmentor::isStaticClass(semantic_label)) {
        return base_threshold * (occluded_map_thr1_static / occluded_map_thr1);
    }
    
    return base_threshold;
}

bool DynObjFilter::Case1(point_soph& p) {
    float occluded_map = 0.0f;
    
    for (size_t map_idx = 0; map_idx < depth_map_list.size(); map_idx++) {
        const auto& depth_map = depth_map_list[map_idx];
        if (fabs(p.time - depth_map->start_time) < frame_dur) continue;
        
        int pos = p.hor_ind * MAX_1D_HALF + p.ver_ind;
        if (pos < 0 || pos >= MAX_2D_N) continue;
        
        const auto& points_in_pixel = depth_map->depth_map[pos];
        for (const auto* other_p : points_in_pixel) {
            if (other_p->dyn == INVALID) continue;
            if (other_p->vec(2) > p.vec(2) + 0.1f) {
                occluded_map += 1.0f;
                break;
            }
        }
    }
    
    return occluded_map >= occluded_map_thr1;
}

bool DynObjFilter::Case1WithSemantic(point_soph& p, int semantic_label, float confidence) {
    float occluded_map = 0.0f;
    float adaptive_thr = getAdaptiveThreshold(semantic_label, confidence, occluded_map_thr1);
    
    for (size_t map_idx = 0; map_idx < depth_map_list.size(); map_idx++) {
        const auto& depth_map = depth_map_list[map_idx];
        if (fabs(p.time - depth_map->start_time) < frame_dur) continue;
        
        int pos = p.hor_ind * MAX_1D_HALF + p.ver_ind;
        if (pos < 0 || pos >= MAX_2D_N) continue;
        
        const auto& points_in_pixel = depth_map->depth_map[pos];
        for (const auto* other_p : points_in_pixel) {
            if (other_p->dyn == INVALID) continue;
            if (other_p->vec(2) > p.vec(2) + 0.1f) {
                occluded_map += 1.0f;
                break;
            }
        }
    }
    
    return occluded_map >= adaptive_thr;
}

bool DynObjFilter::Case2(point_soph& p) {
    return p.occu_times >= occluded_times_thr2;
}

bool DynObjFilter::Case2WithSemantic(point_soph& p, int semantic_label, float confidence) {
    int adaptive_occu_thr = occluded_times_thr2;
    
    if (confidence >= semantic_conf_threshold) {
        if (SemanticSegmentor::isDynamicClass(semantic_label)) {
            adaptive_occu_thr = std::max(1, occluded_times_thr2 - 1);
        } else if (SemanticSegmentor::isStaticClass(semantic_label)) {
            adaptive_occu_thr = occluded_times_thr2 + 1;
        }
    }
    
    return p. occu_times >= adaptive_occu_thr;
}

bool DynObjFilter::Case3(point_soph& p) {
    return p.is_occu_times >= occluding_times_thr3;
}

bool DynObjFilter::Case3WithSemantic(point_soph& p, int semantic_label, float confidence) {
    int adaptive_occuing_thr = occluding_times_thr3;
    
    if (confidence >= semantic_conf_threshold) {
        if (SemanticSegmentor::isDynamicClass(semantic_label)) {
            adaptive_occuing_thr = std::max(1, occluding_times_thr3 - 1);
        } else if (SemanticSegmentor::isStaticClass(semantic_label)) {
            adaptive_occuing_thr = occluding_times_thr3 + 1;
        }
    }
    
    return p.is_occu_times >= adaptive_occuing_thr;
}

void DynObjFilter::buildDepthMap(const std::vector<point_soph*>& points, 
                                DepthMap::Ptr& depth_map) {
    depth_map->clear();
    
    for (auto* p : points) {
        int pos = p->hor_ind * MAX_1D_HALF + p->ver_ind;
        if (pos >= 0 && pos < MAX_2D_N) {
            depth_map->depth_map[pos].push_back(p);
        }
    }
}

void DynObjFilter::updateDepthMapList(double current_time) {
    while (! depth_map_list.empty() && 
           current_time - depth_map_list.front()->end_time > depth_map_dur) {
        depth_map_list.pop_front();
    }
    
    while (depth_map_list.size() > static_cast<size_t>(max_depth_map_num)) {
        depth_map_list. pop_front();
    }
}

void DynObjFilter::filter(PointCloudXYZI::Ptr feats_undistort, 
                         const M3D& rot_end, 
                         const V3D& pos_end, 
                         const double& scan_end_time) {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    if (! dyn_filter_en || feats_undistort->points. empty()) {
        return;
    }
    
    laserCloudDynObj->clear();
    laserCloudDynObj_world->clear();
    laserCloudSteadObj->clear();
    laserCloudDynObj_clus->clear();
    laserCloudSteadObj_clus->clear();
    
    case1_num = case2_num = case3_num = 0;
    
    // Semantic segmentation
    bool has_semantic = false;
    if (use_semantic && semantic_segmentor->isInitialized()) {
        auto t_sem_start = std::chrono::high_resolution_clock::now();
        
        // --- START: 将输入点云转换为 pcl::PointCloud<pcl::PointXYZI>::Ptr ---
        // semantic_segmentor->segment 接受的是 PointXYZI::Ptr，因此做一次拷贝/转换
        pcl::PointCloud<pcl::PointXYZI>::Ptr feats_undistort_i(new pcl::PointCloud<pcl::PointXYZI>());
        feats_undistort_i->header = feats_undistort->header;
        feats_undistort_i->width = feats_undistort->width;
        feats_undistort_i->height = feats_undistort->height;
        feats_undistort_i->is_dense = feats_undistort->is_dense;
        feats_undistort_i->points.reserve(feats_undistort->points.size());
        
        for (const auto& p : feats_undistort->points) {
            pcl::PointXYZI pi;
            pi.x = p.x;
            pi.y = p.y;
            pi.z = p.z;
            // 假设输入点类型包含 intensity 字段（常见于 PointXYZI / PointXYZINormal）
            pi.intensity = p.intensity;
            feats_undistort_i->points.push_back(pi);
        }
        
        // 用转换后的 clouds 调用分割
        has_semantic = semantic_segmentor->segment(feats_undistort_i, 
                                                   current_semantic_labels,
                                                   current_semantic_confidences);
        // --- END: 转换与调用 ---
        
        auto t_sem_end = std::chrono::high_resolution_clock::now();
        double sem_time = std::chrono::duration<double, std::milli>(t_sem_end - t_sem_start). count();
        
        if (! has_semantic) {
            ROS_WARN_THROTTLE(5.0, "Semantic segmentation failed");
        } else {
            ROS_DEBUG("Semantic segmentation: %.2f ms", sem_time);
        }
    }
    
    // Process points
    int size = feats_undistort->points.size();
    std::vector<point_soph*> current_points;
    current_points.reserve(size);
    
    float fov_down_rad = fov_down * M_PI / 180.0f;
    float fov_up_rad = fov_up * M_PI / 180.0f;
    float fov_range = fov_up_rad - fov_down_rad;
    
    for (int i = 0; i < size; i++) {
        if (cur_point_soph_pointers >= max_pointers_num) {
            cur_point_soph_pointers = 0;
        }
        
        point_soph* p = point_soph_pointers[cur_point_soph_pointers++];
        p->glob = V3D(feats_undistort->points[i].x,
                     feats_undistort->points[i].y,
                     feats_undistort->points[i].z);
        p->local = rot_end. transpose() * (p->glob - pos_end);
        p->time = scan_end_time;
        p->dyn = STATIC;
        p->occu_times = 0;
        p->is_occu_times = 0;
        
        float range = p->local.norm();
        if (range < blind_dis) {
            p->dyn = INVALID;
            continue;
        }
        
        float azimuth = atan2(p->local.y(), p->local.x());
        float elevation = asin(p->local. z() / range);
        
        p->vec = V3D(azimuth, elevation, range);
        p->hor_ind = static_cast<int>((azimuth + M_PI) / hor_resolution_max);
        p->ver_ind = static_cast<int>((elevation - fov_down_rad) / ver_resolution_max);
        
        p->hor_ind = std::max(0, std::min(p->hor_ind, MAX_1D - 1));
        p->ver_ind = std::max(0, std::min(p->ver_ind, MAX_1D_HALF - 1));
        
        if (has_semantic && i < current_semantic_labels.size()) {
            p->semantic_label = current_semantic_labels[i];
            p->semantic_confidence = current_semantic_confidences[i];
        } else {
            p->semantic_label = UNLABELED;
            p->semantic_confidence = 0.0f;
        }
        
        current_points.push_back(p);
    }
    
    // Build current depth map
    DepthMap::Ptr current_depth_map = std::make_shared<DepthMap>();
    current_depth_map->start_time = scan_end_time;
    current_depth_map->end_time = scan_end_time + frame_dur;
    buildDepthMap(current_points, current_depth_map);
    
    // Update depth map list
    depth_map_list.push_back(current_depth_map);
    updateDepthMapList(scan_end_time);
    
    // Dynamic detection
    #pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < current_points.size(); i++) {
        point_soph* p = current_points[i];
        if (p->dyn == INVALID) continue;
        
        bool is_dynamic = false;
        
        if (has_semantic) {
            if (Case1WithSemantic(*p, p->semantic_label, p->semantic_confidence)) {
                p->dyn = CASE1;
                #pragma omp atomic
                case1_num++;
                is_dynamic = true;
            } else if (Case2WithSemantic(*p, p->semantic_label, p->semantic_confidence)) {
                p->dyn = CASE2;
                #pragma omp atomic
                case2_num++;
                is_dynamic = true;
            } else if (Case3WithSemantic(*p, p->semantic_label, p->semantic_confidence)) {
                p->dyn = CASE3;
                #pragma omp atomic
                case3_num++;
                is_dynamic = true;
            }
        } else {
            if (Case1(*p)) {
                p->dyn = CASE1;
                #pragma omp atomic
                case1_num++;
                is_dynamic = true;
            } else if (Case2(*p)) {
                p->dyn = CASE2;
                #pragma omp atomic
                case2_num++;
                is_dynamic = true;
            } else if (Case3(*p)) {
                p->dyn = CASE3;
                #pragma omp atomic
                case3_num++;
                is_dynamic = true;
            }
        }
    }
    
    // Collect results
    std::vector<int> dyn_indices;
    for (size_t i = 0; i < current_points.size(); i++) {
        point_soph* p = current_points[i];
        
        PointType po;
        po.x = p->glob(0);
        po.y = p->glob(1);
        po.z = p->glob(2);
        po.intensity = i;
        po.curvature = p->semantic_label;
        po.normal_x = p->dyn;
        po.normal_y = p->occu_times;
        po. normal_z = p->is_occu_times;
        
        if (p->dyn == CASE1 || p->dyn == CASE2 || p->dyn == CASE3) {
            laserCloudDynObj->push_back(po);
            laserCloudDynObj_world->push_back(po);
            dyn_indices.push_back(i);
        } else if (p->dyn == STATIC) {
            laserCloudSteadObj->push_back(po);
        }
    }
    
    // Clustering
    if (cluster_coupled && ! dyn_indices.empty()) {
        std::vector<int> dyn_tag;
        
        if (has_semantic) {
            std::vector<int> dyn_semantic_labels;
            std::vector<float> dyn_confidences;
            
            for (int idx : dyn_indices) {
                dyn_semantic_labels.push_back(current_semantic_labels[idx]);
                dyn_confidences.push_back(current_semantic_confidences[idx]);
            }
            
            Cluster. ClusterprocessWithSemantic(dyn_tag, *laserCloudDynObj, 
                                              dyn_semantic_labels, dyn_confidences,
                                              *laserCloudDynObj_world, 
                                              std_msgs::Header(), rot_end, pos_end);
        } else {
            Cluster. Clusterprocess(dyn_tag, *laserCloudDynObj, *laserCloudDynObj_world,
                                  std_msgs::Header(), rot_end, pos_end);
        }
        
        // Filter based on clustering
        for (size_t i = 0; i < dyn_tag.size(); i++) {
            if (dyn_tag[i] > 0) {
                laserCloudDynObj_clus->push_back(laserCloudDynObj->points[i]);
            } else {
                laserCloudSteadObj_clus->push_back(laserCloudDynObj->points[i]);
            }
        }
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    time_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    ROS_DEBUG("Filter time: %.2f ms, Case1: %d, Case2: %d, Case3: %d", 
              time_total, case1_num, case2_num, case3_num);
}

void DynObjFilter::publish_dyn(const ros::Publisher& pub_point_out, 
                              const ros::Publisher& pub_frame_out, 
                              const ros::Publisher& pub_steady_points, 
                              const double& scan_end_time) {
    if (cluster_coupled) {
        ROS_INFO("Dynamic objects: %lu, Total time: %.2f ms", 
                 laserCloudDynObj_clus->points.size(), time_total);
    } else {
        ROS_INFO("Dynamic objects: %lu, Total time: %.2f ms", 
                 laserCloudDynObj->points.size(), time_total);
    }
    
    ROS_INFO("Case1: %d, Case2: %d, Case3: %d", case1_num, case2_num, case3_num);
    
    sensor_msgs::PointCloud2 msg_dyn;
    pcl::toROSMsg(*laserCloudDynObj_world, msg_dyn);
    msg_dyn.header.stamp = ros::Time(). fromSec(scan_end_time);
    msg_dyn. header.frame_id = frame_id;
    pub_point_out.publish(msg_dyn);
    
    if (cluster_coupled || cluster_future) {
        sensor_msgs::PointCloud2 msg_clus;
        pcl::toROSMsg(*laserCloudDynObj_clus, msg_clus);
        msg_clus.header.stamp = ros::Time().fromSec(scan_end_time);
        msg_clus.header. frame_id = frame_id;
        pub_frame_out. publish(msg_clus);
    }
    
    sensor_msgs::PointCloud2 msg_steady;
    pcl::toROSMsg(*laserCloudSteadObj, msg_steady);
    msg_steady.header.stamp = ros::Time().fromSec(scan_end_time);
    msg_steady.header.frame_id = frame_id;
    pub_steady_points.publish(msg_steady);
}

void DynObjFilter::set_path(string file_path, string file_path_origin) {
    is_set_path = true;
    out_file = file_path;
    out_file_origin = file_path_origin;
}