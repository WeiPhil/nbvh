
#pragma once

#include "base/accel/bvh.h"

namespace ntwr {
struct BVH8Converter {
public:
    BVH8 &m_bvh8;
    const BVH2 &m_cpu_bvh;
    int m_max_depth;

    BVH8Converter(BVH8 &bvh8, const BVH2 &cpu_bvh, int max_depth, const BVH8Converter &bvh_converter)
        : m_bvh8(bvh8), m_cpu_bvh(cpu_bvh), m_max_depth(max_depth)
    {
        m_bvh8.m_indices.reserve(cpu_bvh.m_indices.size());
        m_bvh8.m_nodes.reserve(cpu_bvh.m_nodes.size());
        decisions = bvh_converter.decisions;
    }

    BVH8Converter(BVH8 &bvh8, const BVH2 &cpu_bvh, int max_depth = -1)
        : m_bvh8(bvh8), m_cpu_bvh(cpu_bvh), m_max_depth(max_depth)
    {
        m_bvh8.m_indices.reserve(cpu_bvh.m_indices.size());
        m_bvh8.m_nodes.reserve(cpu_bvh.m_nodes.size());
    }

    void convert(std::vector<BVH2Node> *leaf_nodes = nullptr);

private:
    struct Decision {
        enum struct Type : char { LEAF, INTERNAL, DISTRIBUTE } type;

        char distribute_left;
        char distribute_right;

        float cost;
    };

    std::vector<Decision> decisions;

    int calculate_cost(int node_index, const std::vector<BVH2Node> &nodes);

    void get_children(int node_index, const std::vector<BVH2Node> &nodes, int children[8], int &child_count, int i);
    void order_children(int node_index, const std::vector<BVH2Node> &nodes, int children[8], int child_count);

    int count_primitives(int node_index, const std::vector<BVH2Node> &nodes, const std::vector<size_t> &indices_sbvh);

    void collapse(const std::vector<BVH2Node> &nodes_bvh,
                  const std::vector<size_t> &indices_bvh,
                  int node_index_bvh8,
                  int node_index_bvh2,
                  int depth,
                  std::vector<BVH2Node> *leaf_nodes = nullptr);
};

}
