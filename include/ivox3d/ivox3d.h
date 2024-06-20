#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>

#include <algorithm>
#include <execution>
#include <iterator>
#include <unordered_map>
#include <list>
#include <vector>

template <typename PointT>
class IVoxNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct DistPoint {
        double dist = 0;
        IVoxNode* node = nullptr;
        int idx = 0;

        DistPoint() = default;
        DistPoint(const double d, IVoxNode* n, const int i) : dist(d), node(n), idx(i) {}

        PointT Get() { return node->GetPoint(idx); }

        inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }

        inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
    };

    IVoxNode() = default;
    IVoxNode(const PointT& center, const float& side_length) {}  /// same with phc

    void InsertPoint(const PointT& pt) { points_.template emplace_back(pt); };

    inline bool Empty() const { return points_.empty(); };

    inline std::size_t Size() const { return points_.size(); };

    inline PointT GetPoint(const std::size_t idx) const { return points_[idx]; };

    // 遍历所有的最近栅格，将最近点存入 dis_points，所以存在 old_size
    int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K, const double& max_range) {
        std::size_t old_size = dis_points.size();
        // 遍历体素内的所有点
        for (const auto& pt : points_) {
            // 计算遍历点和当前点的距离^2
            double d = (pt.getVector3fMap() - point.getVector3fMap()).squaredNorm();
            // 如果距离低于最大距离阈值，则将其放入候选近邻点队列中
            if (d < max_range * max_range) {
                dis_points.template emplace_back(DistPoint(d, this, &pt - points_.data()));
            }
        }

        // sort by distance
        if (old_size + K < dis_points.size()) {
            // 按距离排序，并取前面 max_num*nearby_grids_.size()+max_num 个近邻点
            std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1, dis_points.end());
            dis_points.resize(old_size + K);
        }

        return dis_points.size();
    }

private:
    std::vector<PointT> points_;
};

template <typename PointType = pcl::PointXYZ>
class IVox {
public:
    struct hash_vec {
        inline size_t operator()(const Eigen::Matrix<int, 3, 1>& v) const {
            return size_t(((v[0]) * 73856093) ^ ((v[1]) * 471943) ^ ((v[2]) * 83492791)) % 10000000;
        };
    };

    using KeyType = Eigen::Vector3i;
    using NodeType = IVoxNode<PointType>;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using DistPoint = typename NodeType::DistPoint;

    // 用户决定的邻近体素个数
    enum class NearbyType {
        CENTER,  // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    // 存储体素的参数
    struct Options {
        float resolution_ = 0.2;                        // ivox resolution
        float inv_resolution_ = 10.0;                   // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // nearby range
        std::size_t capacity_ = 1000000;                // capacity
    };

    // 初始化一个体素后会生成近邻体素
    explicit IVox(Options options) : options_(options) {
        options_.inv_resolution_ = 1.0 / options_.resolution_;  // 配置体素的分辨率
        GenerateNearbyGrids();
    }

    void AddPoints(const PointVector& points_to_add);

    /// get nn
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt);

    /// get nn with condition
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num = 5, double max_range = 5.0);

    /// get nn in cloud
    bool GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud);

    /// get number of valid grids
    size_t NumValidGrids() const { return grids_map_.size(); };

    /// get statistics of the points
    std::vector<float> StatGridPoints() const;

private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    // Matrix和Vector即矩阵和向量，和数学上一致。 .array() 非数学形式逐个计算  .round() 四舍五入
    KeyType Pos2Grid(const PointType& pt) const { return (pt.getVector3fMap() * options_.inv_resolution_).array().round().template cast<int>(); };

    Options options_;

    // LRU实现
    // KeyType：表示哈希表 键的类型, typename list<pair<KeyType, NodeType>>::iterator 表示哈希表中值的类型
    // 值的类型是指向 list 中元素的迭代器, 指list中的一个元素，而不是整个list。
    // list 的元素类型是一个由 KeyType 和 NodeType 构成的 pair, 整个list的元素构成整个体素地图
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, hash_vec> grids_map_;  // voxel hash map
    // grids_cache_是list双向链表，存储有序的元素。。链表中的每个元素都是一个由KeyType和NodeType组成的键值对，表示一个网格及其在哈希表中的键值。
    // 哈希表 grids_map_ 提供了高效的键值查找功能，而链表 grids_cache_ 存储了实际的网格数据。在保持数据有序的同时，实现高效的查找和存储操作
    std::list<std::pair<KeyType, NodeType>> grids_cache_;  // voxel cache
    std::vector<KeyType> nearby_grids_;                    // nearbys
};

template <typename PointType>
bool IVox<PointType>::GetClosestPoint(const PointType& pt, PointType& closest_pt) {
    std::vector<DistPoint> candidates;
    auto key = Pos2Grid(pt);
    std::for_each(nearby_grids_.begin(), nearby_grids_.end(), [&key, &candidates, &pt, this](const KeyType& delta) {
        auto dkey = key + delta;
        auto iter = grids_map_.find(dkey);
        if (iter != grids_map_.end()) {
            DistPoint dist_point;
            bool found = iter->second->second.NNPoint(pt, dist_point);
            if (found) candidates.emplace_back(dist_point);
        }
    });

    if (candidates.empty()) return false;

    auto iter = std::min_element(candidates.begin(), candidates.end());
    closest_pt = iter->Get();
    return true;
}

template <typename PointType>
bool IVox<PointType>::GetClosestPoint(const PointType& pt, PointVector& closest_pt, int max_num, double max_range) {
    // 候选近邻点，每个点都有对应的距离
    std::vector<DistPoint> candidates;
    candidates.reserve(max_num * nearby_grids_.size());
    // 计算该点所属体素的索引
    auto key = Pos2Grid(pt);
    // 遍历近邻的体素， nearby_grids_存储的是邻近体素的相对偏移
    for (const KeyType& delta : nearby_grids_) {
        auto dkey = key + delta;            // 在地图中的实际体素索引
        auto iter = grids_map_.find(dkey);  // 找到对应的体素
        if (iter != grids_map_.end()) {
            auto tmp = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
        }
    }

    // 如果候选点队列为空，找不到近邻点
    if (candidates.empty()) return false;

    // 需要根据距离进行选择
    if (candidates.size() > max_num) {
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        candidates.resize(max_num);  // 直接通过resize截除后半段距离太远的点
    }
    // 进一步，将距离最小的点挪到第一位
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

    // 将近邻点放进去closest_pt
    closest_pt.clear();
    for (auto& it : candidates) closest_pt.emplace_back(it.Get());

    return closest_pt.empty() == false;
}

// 为当前体素生成近邻体素坐标的相对偏移量
template <typename PointType>
void IVox<PointType>::GenerateNearbyGrids() {
    // 参数文件中 ivox_nearby_type: 18   # 6, 18, 26
    if (options_.nearby_type_ == NearbyType::CENTER) {
        nearby_grids_.emplace_back(KeyType::Zero());
    } else if (options_.nearby_type_ == NearbyType::NEARBY6) {
        nearby_grids_ = {KeyType(0, 0, 0), KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0), KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY18) {
        nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0), KeyType(1, 0, 0),  KeyType(0, 1, 0),   KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),
                         KeyType(1, 1, 0),   KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),  KeyType(-1, 0, 1), KeyType(1, 0, -1),
                         KeyType(-1, 0, -1), KeyType(0, 1, 1),  KeyType(0, -1, 1), KeyType(0, 1, -1),  KeyType(0, -1, -1)};
    } else if (options_.nearby_type_ == NearbyType::NEARBY26) {
        nearby_grids_ = {KeyType(0, 0, 0),   KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),  KeyType(0, -1, 0),  KeyType(0, 0, -1),
                         KeyType(0, 0, 1),   KeyType(1, 1, 0),   KeyType(-1, 1, 0),  KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
                         KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),  KeyType(0, -1, 1),  KeyType(0, 1, -1),
                         KeyType(0, -1, -1), KeyType(1, 1, 1),   KeyType(-1, 1, 1),  KeyType(1, -1, 1), KeyType(1, 1, -1),  KeyType(-1, -1, 1),
                         KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
    } else {
        std::cerr << "Unknown nearby_type!";
    }
}

template <typename PointType>
bool IVox<PointType>::GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) {
    // 给点云的每个点分配索引
    std::vector<size_t> index(cloud.size());
    for (int i = 0; i < cloud.size(); ++i) index[i] = i;
    closest_cloud.resize(cloud.size());
    // 多线程遍历点云
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&cloud, &closest_cloud, this](size_t idx) {
        PointType pt;
        if (GetClosestPoint(cloud[idx], pt))
            closest_cloud[idx] = pt;
        else
            closest_cloud[idx] = PointType();
    });
    return true;
}

template <typename PointType>
void IVox<PointType>::AddPoints(const PointVector& points_to_add) {
    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), [this](const auto& pt) {
        // 将点的坐标转成hash值索引(3维的)
        auto key = Pos2Grid(pt);
        // 在栅格地图中寻找是否有对应的栅格
        auto iter = grids_map_.find(key);  // iter->second->second : NodeType/IVoxNode
        // 没有对应的栅格的话就要新建栅格
        if (iter == grids_map_.end()) {
            PointType center;
            // getVector3fMap() : pcl::PointXYZ将转成Eigen::Vector3f
            // 计算新的体素中心坐标，也就是新加入的点的坐标
            center.getVector3fMap() = key.template cast<float>() * options_.resolution_;
            // 在栅格地图缓存中加入新的体素
            grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
            // 让key和对应体素绑定
            grids_map_.insert({key, grids_cache_.begin()});
            // 向该体素中加入点
            grids_cache_.front().second.InsertPoint(pt);
            // 如果地图的总体素个数大于阈值，则要将旧的体素删除，如论文Fig.4
            if (grids_map_.size() >= options_.capacity_) {
                grids_map_.erase(grids_cache_.back().first);
                grids_cache_.pop_back();
            }
            // 寻找到已有栅格
        } else {
            // 每个体素里通过vector存放点，InsertPoint就是emplace_back
            iter->second->second.InsertPoint(pt);
            // 缓存拼接, https://blog.csdn.net/boiled_water123/article/details/103753598中的方法二
            // 将iter->second剪接到grids_cache_.begin()的位置，维护体素地图中的体素由新到旧进行排列
            // 对应论文中的Fig.4
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            // 重新让key和对应体素绑定
            grids_map_[key] = grids_cache_.begin();
        }
    });
}

template <typename PointType>
std::vector<float> IVox<PointType>::StatGridPoints() const {
    int num = grids_cache_.size(), valid_num = 0, max = 0, min = 100000000;
    int sum = 0, sum_square = 0;
    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        valid_num += s > 0;
        max = s > max ? s : max;
        min = s < min ? s : min;
        sum += s;
        sum_square += s * s;
    }
    float ave = float(sum) / num;
    float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;
    return std::vector<float>{(float)valid_num, ave, (float)max, (float)min, stddev};
}
