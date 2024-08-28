#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

#include <algorithm>
#include <execution>
#include <iterator>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <list>

enum EstiPlane { None = 0, Prior, Cur, PriorE };

template <typename PointType>
class HVoxNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // using CloudType = pcl::PointCloud<PointType>;

    struct DistPoint {
        double dist = 0;
        PointType pt;

        DistPoint() = default;
        DistPoint(const double d, PointType pt) : dist(d), pt(pt) {}

        const PointType& Get() { return pt; }

        inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }
        inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
    };

    HVoxNode() = default;
    HVoxNode(const PointType& pt) { InsertPoint(pt); }

    void InsertPoint(const PointType& pt) { points_.template emplace_back(pt); };

    inline PointType GetPoint(const std::size_t idx) const { return points_[idx]; };

    inline bool empty() const { return points_.empty(); };

    inline std::size_t size() const { return points_.size(); };

    inline void clear() { points_.clear(); }

    int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointType& point, const int& K, const double& max_range) {
        std::size_t old_size = dis_points.size();

        for (const auto& pt : points_) {
            double d = (pt.getVector3fMap() - point.getVector3fMap()).squaredNorm();
            if (d < max_range * max_range) dis_points.template emplace_back(DistPoint(d, pt));
        }

        // sort by distance
        if (old_size + K < dis_points.size()) {
            std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1, dis_points.end());
            dis_points.resize(old_size + K);
        }

        return dis_points.size();
    }

    int curprob = 10;

private:
    std::vector<PointType> points_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename PointType>
class HVox {
public:
    struct hash_vec {
        inline size_t operator()(const Eigen::Matrix<int, 3, 1>& v) const {
            return size_t(((v[0]) * 73856093) ^ ((v[1]) * 471943) ^ ((v[2]) * 83492791)) % 10000000;
        };
    };

    using KeyType = Eigen::Vector3i;
    using KeyVec = std::vector<KeyType>;
    using KeySet = std::set<KeyType>;
    using KeyUSet = std::unordered_set<KeyType, hash_vec>;

    using NodeType = HVoxNode<PointType>;
    using DistPoint = typename NodeType::DistPoint;

    using CloudType = pcl::PointCloud<PointType>;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

    enum class NearbyType {
        CENTER,
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    struct Options {
        float res_ = 0.2;                               // ivox resolution
        float inv_res_ = 10.0;                          // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // nearby range
        std::size_t capacity_ = 1000000;                // capacity
    };

    std::unordered_map<KeyType, NodeType, hash_vec> grids_map_;
    KeyVec nearby_grids_;
    Options opt_;

    // explicit
    HVox(Options options) : opt_(options) {
        opt_.inv_res_ = 1.0 / opt_.res_;
        GenerateNearbyGrids();
    }

    HVox() {
        opt_.inv_res_ = 1.0 / opt_.res_;
        GenerateNearbyGrids();
    }

    KeyType Pos2Grid(const PointType& pt) const { return (pt.getVector3fMap() * opt_.inv_res_).array().round().template cast<int>(); };

    size_t NumValidGrids() const { return grids_map_.size(); };

    template <typename Container>  //  CloudType PointVector
    void AddPoints(const Container& points_to_add) {
        std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const auto& pt) {
            KeyType key = Pos2Grid(pt);
            auto iter = grids_map_.find(key);
            if (iter == grids_map_.end()) {
                grids_map_[key] = NodeType(pt);
            } else {
                iter->second.InsertPoint(pt);
            }
        });
    }

    // template <typename Container>
    void UpdateProb(const CloudType& points, const std::vector<EstiPlane>& point_selected_surf, const std::vector<PointVector>& Nearest_Points,
                    CloudType& point_out) {
        point_out.clear();
        int curcnt = 0;

        for (int i = 0; i < points.size(); i++) {
            if (point_selected_surf[i] == EstiPlane::Prior) continue;  // 先验关联上
            auto& pt = points[i];
            const PointVector& points_near = Nearest_Points[i];

            KeyType key = Pos2Grid(pt);

            auto iter = grids_map_.find(key);

            if (iter != grids_map_.end()) {
                NodeType& node = iter->second;

                // 点数太少
                if (points_near.size() < 5) {
                    node.InsertPoint(pt);
                    point_out.push_back(pt);
                } else {
                    // 最近点也很远
                    Eigen::Vector3f center = ((pt.getVector3fMap() / opt_.res_).array().floor() + 0.5) * opt_.res_;
                    Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;
                    if (fabs(dis_2_center.x()) > 0.5 * opt_.res_ && fabs(dis_2_center.y()) > 0.5 * opt_.res_ && fabs(dis_2_center.z()) > 0.5 * opt_.res_) {
                        node.InsertPoint(pt);
                        point_out.push_back(pt);
                    }

                    // for (int i = 0; i < 5; i++) {
                    //     if (calc_dist(points_near[i].getVector3fMap(), center) < dist + 1e-6) {
                    //         need_add = false;
                    //         break;
                    //     }
                    // }
                }

                // if (node.size() < 10) node.InsertPoint(pt);

                if (point_selected_surf[i] == EstiPlane::Cur) node.curprob += 100;
            } else {
                grids_map_[key] = NodeType(pt);
            }
        }

        // std::cout << "cur: " << curcnt << " grids bfr: " << grids_map_.size();
        for (auto iter = grids_map_.begin(); iter != grids_map_.end();) {
            NodeType& node = iter->second;

            node.curprob--;
            if (node.curprob < -10)
                iter = grids_map_.erase(iter);
            else
                ++iter;
        }
        // std::cout << " aft: " << grids_map_.size() << std::endl;
    }

    void GetPoints(CloudType& cloud_out, bool dense = false) {
        cloud_out.clear();
        for (auto iter = grids_map_.begin(); iter != grids_map_.end(); ++iter) {
            NodeType& node = iter->second;
            cloud_out.emplace_back(node.GetPoint(0));
        }
    }

    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 5.0) {
        std::vector<DistPoint> candidates;
        candidates.reserve(max_num * nearby_grids_.size());

        KeyType key = Pos2Grid(pt);
        for (const KeyType& delta : nearby_grids_) {
            KeyType dkey = key + delta;
            auto iter = grids_map_.find(dkey);
            if (iter != grids_map_.end()) iter->second.KNNPointByCondition(candidates, pt, max_num, max_range);
        }

        if (candidates.empty()) return false;

        if (candidates.size() > max_num) {
            std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
            candidates.resize(max_num);
        }

        std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

        closest_pt.clear();
        for (auto& it : candidates) closest_pt.emplace_back(it.Get());

        return closest_pt.size();
    }

    void GenerateNearbyGrids() {
        if (opt_.nearby_type_ == NearbyType::CENTER) {
            nearby_grids_.emplace_back(KeyType::Zero());
        } else if (opt_.nearby_type_ == NearbyType::NEARBY6) {
            nearby_grids_ = {KeyType(0, 0, 0), KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0), KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
        } else if (opt_.nearby_type_ == NearbyType::NEARBY18) {
            nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0), KeyType(1, 0, 0),  KeyType(0, 1, 0),  KeyType(0, -1, 0),
                             KeyType(0, 0, -1),  KeyType(0, 0, 1),  KeyType(1, 1, 0),  KeyType(-1, 1, 0), KeyType(1, -1, 0),
                             KeyType(-1, -1, 0), KeyType(1, 0, 1),  KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1),
                             KeyType(0, 1, 1),   KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1)};
        } else if (opt_.nearby_type_ == NearbyType::NEARBY26) {
            nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),  KeyType(0, -1, 0),  KeyType(0, 0, -1),
                             KeyType(0, 0, 1),   KeyType(1, 1, 0),   KeyType(-1, 1, 0),  KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
                             KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),  KeyType(0, -1, 1),  KeyType(0, 1, -1),
                             KeyType(0, -1, -1), KeyType(1, 1, 1),   KeyType(-1, 1, 1),  KeyType(1, -1, 1), KeyType(1, 1, -1),  KeyType(-1, -1, 1),
                             KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
        } else {
            std::cerr << "Unknown nearby_type!";
        }
    }
};
