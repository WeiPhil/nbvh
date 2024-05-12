#include "base/accel/bvh8_converter.h"
#include <iostream>
#include "utils/logger.h"

namespace ntwr {

void BVH8Converter::convert(std::vector<BVH2Node> *leaf_nodes)
{
    m_bvh8.m_indices.clear();
    m_bvh8.m_indices.reserve(m_cpu_bvh.m_indices.size());

    m_bvh8.m_nodes.clear();
    m_bvh8.m_nodes.emplace_back();  // Root

    if (decisions.size() > 0) {
        assert(decisions.size() == m_cpu_bvh.m_nodes.size() * 7);
        logger(LogLevel::Info, "Using precomputed costs");
    } else {
        decisions.resize(m_cpu_bvh.m_nodes.size() * 7);

        // Fill cost table using dynamic programming (bottom up)
        calculate_cost(0, m_cpu_bvh.m_nodes);
    }

    // Collapse SBVH into 8-way tree (top down)
    collapse(m_cpu_bvh.m_nodes, m_cpu_bvh.m_indices, 0, 0, 1, leaf_nodes);
    assert(m_bvh8.m_indices.size() == m_cpu_bvh.m_indices.size());
}

int BVH8Converter::calculate_cost(int node_index, const std::vector<BVH2Node> &nodes)
{
    const BVH2Node &node = nodes[node_index];

    int num_primitives;

    if (node.is_leaf()) {
        num_primitives = node.prim_count;
        if (num_primitives != 1) {
            logger(LogLevel::Error, "BVH8 Builder expects BVH with leaf Nodes containing only 1 primitive!");
            exit(1);
        }

        // SAH cost
        float cost_leaf = node.bbox().area() * float(num_primitives);

        for (int i = 0; i < 7; i++) {
            decisions[node_index * 7 + i].type = Decision::Type::LEAF;
            decisions[node_index * 7 + i].cost = cost_leaf;
        }
    } else {
        num_primitives =
            calculate_cost(node.left_or_prim_offset, nodes) + calculate_cost(node.left_or_prim_offset + 1, nodes);

        // Separate case: i=0 (i=1 in the paper)
        {
            float cost_leaf = num_primitives <= 3 ? node.bbox().area() * float(num_primitives) : FLT_MAX;

            float cost_distribute = FLT_MAX;

            char distribute_left  = INVALID_NODE;
            char distribute_right = INVALID_NODE;

            for (int k = 0; k < 7; k++) {
                float c = decisions[(node.left_or_prim_offset) * 7 + k].cost +
                          decisions[(node.left_or_prim_offset + 1) * 7 + 6 - k].cost;

                if (c < cost_distribute) {
                    cost_distribute = c;

                    distribute_left  = k;
                    distribute_right = 6 - k;
                }
            }

            float cost_internal = cost_distribute + node.bbox().area();

            if (cost_leaf < cost_internal) {
                decisions[node_index * 7].type = Decision::Type::LEAF;
                decisions[node_index * 7].cost = cost_leaf;
            } else {
                decisions[node_index * 7].type = Decision::Type::INTERNAL;
                decisions[node_index * 7].cost = cost_internal;
            }

            decisions[node_index * 7].distribute_left  = distribute_left;
            decisions[node_index * 7].distribute_right = distribute_right;
        }

        // In the paper i=2..7
        for (int i = 1; i < 7; i++) {
            float cost_distribute = decisions[node_index * 7 + i - 1].cost;

            char distribute_left  = INVALID_NODE;
            char distribute_right = INVALID_NODE;

            for (int k = 0; k < i; k++) {
                float c = decisions[(node.left_or_prim_offset) * 7 + k].cost +
                          decisions[(node.left_or_prim_offset + 1) * 7 + i - k - 1].cost;

                if (c < cost_distribute) {
                    cost_distribute = c;

                    distribute_left  = k;
                    distribute_right = i - k - 1;
                }
            }

            decisions[node_index * 7 + i].cost = cost_distribute;

            if (distribute_left != INVALID_NODE) {
                decisions[node_index * 7 + i].type             = Decision::Type::DISTRIBUTE;
                decisions[node_index * 7 + i].distribute_left  = distribute_left;
                decisions[node_index * 7 + i].distribute_right = distribute_right;
            } else {
                decisions[node_index * 7 + i] = decisions[node_index * 7 + i - 1];
            }
        }
    }

    return num_primitives;
}

void BVH8Converter::get_children(
    int node_index, const std::vector<BVH2Node> &nodes, int children[8], int &child_count, int i)
{
    const BVH2Node &node = nodes[node_index];

    if (node.is_leaf()) {
        children[child_count++] = node_index;
        return;
    }

    char distribute_left  = decisions[node_index * 7 + i].distribute_left;
    char distribute_right = decisions[node_index * 7 + i].distribute_right;

    assert(distribute_left >= 0 && distribute_left < 7);
    assert(distribute_right >= 0 && distribute_right < 7);

    // Recurse on left child if it needs to distribute
    if (decisions[node.left_or_prim_offset * 7 + distribute_left].type == Decision::Type::DISTRIBUTE) {
        get_children(node.left_or_prim_offset, nodes, children, child_count, distribute_left);
    } else {
        children[child_count++] = node.left_or_prim_offset;
    }

    // Recurse on right child if it needs to distribute
    if (decisions[(node.left_or_prim_offset + 1) * 7 + distribute_right].type == Decision::Type::DISTRIBUTE) {
        get_children(node.left_or_prim_offset + 1, nodes, children, child_count, distribute_right);
    } else {
        children[child_count++] = node.left_or_prim_offset + 1;
    }
}

void BVH8Converter::order_children(int node_index, const std::vector<BVH2Node> &nodes, int children[8], int child_count)
{
    glm::vec3 p = nodes[node_index].bbox().center();

    float cost[8][8] = {};

    // Fill cost table
    for (int c = 0; c < child_count; c++) {
        for (int s = 0; s < 8; s++) {
            glm::vec3 direction((s & 0b100) ? -1.0f : +1.0f, (s & 0b010) ? -1.0f : +1.0f, (s & 0b001) ? -1.0f : +1.0f);
            glm::vec3 child_bbox_center = nodes[children[c]].bbox().center();

            cost[c][s] = glm::dot(child_bbox_center - p, direction);
        }
    }

    int assignment[8] = {
        INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE};
    bool slot_filled[8] = {};

    // Greedy child ordering, the paper mentions this as an alternative
    // that works about as well as the auction algorithm in practice
    while (true) {
        float min_cost = FLT_MAX;

        int min_slot  = INVALID_NODE;
        int min_index = INVALID_NODE;

        // Find cheapest unfilled slot of any unassigned child
        for (int c = 0; c < child_count; c++) {
            if (assignment[c] == INVALID_NODE) {
                for (int s = 0; s < 8; s++) {
                    if (!slot_filled[s] && cost[c][s] < min_cost) {
                        min_cost = cost[c][s];

                        min_slot  = s;
                        min_index = c;
                    }
                }
            }
        }

        if (min_slot == INVALID_NODE)
            break;

        slot_filled[min_slot] = true;
        assignment[min_index] = min_slot;
    }

    // Permute children array according to assignment
    int children_copy[8] = {};
    for (int i = 0; i < 8; i++) {
        children_copy[i] = children[i];
        children[i]      = INVALID_NODE;
    }
    for (int i = 0; i < child_count; i++) {
        assert(assignment[i] != INVALID_NODE);
        assert(children_copy[i] != INVALID_NODE);
        children[assignment[i]] = children_copy[i];
    }
}

// Recursively count triangles in subtree of the given Node
// Simultaneously fills the indices buffer of the BVH8
int BVH8Converter::count_primitives(int node_index,
                                    const std::vector<BVH2Node> &nodes,
                                    const std::vector<size_t> &indices)
{
    const BVH2Node &node = nodes[node_index];

    if (node.is_leaf()) {
        assert(node.prim_count == 1);

        for (uint32_t i = 0; i < node.prim_count; i++) {
            m_bvh8.m_indices.push_back(indices[node.left_or_prim_offset + i]);
        }

        return node.prim_count;
    }

    return count_primitives(node.left_or_prim_offset, nodes, indices) +
           count_primitives(node.left_or_prim_offset + 1, nodes, indices);
}

Bbox3f quantized_bbox_to_world_bbox(const BVH8Node &node, uint32_t child_idx)
{
    float ex = glm::uintBitsToFloat(node.e[0] << 23);
    float ey = glm::uintBitsToFloat(node.e[1] << 23);
    float ez = glm::uintBitsToFloat(node.e[2] << 23);

    float bbox_min_x = node.p.x + ex * float(node.quantized_min_x[child_idx]);
    float bbox_min_y = node.p.y + ey * float(node.quantized_min_y[child_idx]);
    float bbox_min_z = node.p.z + ez * float(node.quantized_min_z[child_idx]);

    float bbox_max_x = node.p.x + ex * float(node.quantized_max_x[child_idx]);
    float bbox_max_y = node.p.y + ey * float(node.quantized_max_y[child_idx]);
    float bbox_max_z = node.p.z + ez * float(node.quantized_max_z[child_idx]);

    return Bbox3f{glm::vec3(bbox_min_x, bbox_min_y, bbox_min_z), glm::vec3(bbox_max_x, bbox_max_y, bbox_max_z)};
}

void BVH8Converter::collapse(const std::vector<BVH2Node> &nodes_bvh,
                             const std::vector<size_t> &indices_bvh,
                             int node_index_bvh8,
                             int node_index_bvh2,
                             int depth,
                             std::vector<BVH2Node> *leaf_nodes)
{
    BVH8Node &node     = m_bvh8.m_nodes[node_index_bvh8];
    const Bbox3f &bbox = nodes_bvh[node_index_bvh2].bbox();

    node.p = bbox.min;

    constexpr int Nq      = 8;
    constexpr float denom = 1.0f / float((1 << Nq) - 1);

    glm::vec3 e(exp2f(ceilf(log2f((bbox.max[0] - bbox.min[0]) * denom))),
                exp2f(ceilf(log2f((bbox.max[1] - bbox.min[1]) * denom))),
                exp2f(ceilf(log2f((bbox.max[2] - bbox.min[2]) * denom))));

    glm::vec3 one_over_e(1.0f / e.x, 1.0f / e.y, 1.0f / e.z);

    uint32_t u_ex = glm::floatBitsToUint(e.x);
    uint32_t u_ey = glm::floatBitsToUint(e.y);
    uint32_t u_ez = glm::floatBitsToUint(e.z);

    // Only the exponent bits can be non-zero
    assert((u_ex & 0b10000000011111111111111111111111) == 0);
    assert((u_ey & 0b10000000011111111111111111111111) == 0);
    assert((u_ez & 0b10000000011111111111111111111111) == 0);

    // Store only 8 bit exponent
    node.e[0] = u_ex >> 23;
    node.e[1] = u_ey >> 23;
    node.e[2] = u_ez >> 23;

    int child_count = 0;
    int children[8] = {
        INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE, INVALID_NODE};

    if (m_max_depth == 0) {
        // Special case where only the root should be built
        // To simulate this we create an interior node with a single leaf child
        const Bbox3f &root_bbox = nodes_bvh[0].bbox();

        node.quantized_min_x[0] = byte(floorf((root_bbox.min[0] - node.p.x) * one_over_e.x));
        node.quantized_min_y[0] = byte(floorf((root_bbox.min[1] - node.p.y) * one_over_e.y));
        node.quantized_min_z[0] = byte(floorf((root_bbox.min[2] - node.p.z) * one_over_e.z));

        node.quantized_max_x[0] = byte(ceilf((root_bbox.max[0] - node.p.x) * one_over_e.x));
        node.quantized_max_y[0] = byte(ceilf((root_bbox.max[1] - node.p.y) * one_over_e.y));
        node.quantized_max_z[0] = byte(ceilf((root_bbox.max[2] - node.p.z) * one_over_e.z));

        count_primitives(0, nodes_bvh, indices_bvh);

        // Three highest bits contain unary representation of triangle count
        for (int j = 0; j < 1; j++) {
            node.meta[0] |= (1 << (j + 5));
        }
        node.meta[0] |= 1;

        if (leaf_nodes)
            leaf_nodes->push_back(nodes_bvh[0]);
        return;
    }

    get_children(node_index_bvh2, nodes_bvh, children, child_count, 0);
    assert(child_count <= 8);

    order_children(node_index_bvh2, nodes_bvh, children, child_count);

    node.imask = 0;

    node.base_index_triangle = (uint32_t)m_bvh8.m_indices.size();
    node.base_index_child    = (uint32_t)m_bvh8.m_nodes.size();

    int num_internal_nodes = 0;
    int num_triangles      = 0;

    // If we reached the maximum depth force a leaf decision type and terminate recursion
    bool early_leaf_node_exit = m_max_depth != -1 && depth == m_max_depth;

    for (int i = 0; i < 8; i++) {
        int child_index = children[i];
        if (child_index == INVALID_NODE)
            continue;  // Empty slot

        const Bbox3f &child_bbox = nodes_bvh[child_index].bbox();

        node.quantized_min_x[i] = byte(floorf((child_bbox.min[0] - node.p.x) * one_over_e.x));
        node.quantized_min_y[i] = byte(floorf((child_bbox.min[1] - node.p.y) * one_over_e.y));
        node.quantized_min_z[i] = byte(floorf((child_bbox.min[2] - node.p.z) * one_over_e.z));

        node.quantized_max_x[i] = byte(ceilf((child_bbox.max[0] - node.p.x) * one_over_e.x));
        node.quantized_max_y[i] = byte(ceilf((child_bbox.max[1] - node.p.y) * one_over_e.y));
        node.quantized_max_z[i] = byte(ceilf((child_bbox.max[2] - node.p.z) * one_over_e.z));

// #define BBOX_QUANTIZATION_ERROR
#ifdef BBOX_QUANTIZATION_ERROR
        auto bbox_child = quantized_bbox_to_world_bbox(node, i);

        printf("before child_bbox(%f,%f,%f)-(%f,%f,%f) \n",
               child_bbox.min[0],
               child_bbox.min[1],
               child_bbox.min[2],
               child_bbox.max[0],
               child_bbox.max[1],
               child_bbox.max[2]);

        printf("after child_bbox(%f,%f,%f)-(%f,%f,%f) \n",
               bbox_child.min[0],
               bbox_child.min[1],
               bbox_child.min[2],
               bbox_child.max[0],
               bbox_child.max[1],
               bbox_child.max[2]);
#endif

        Decision::Type decision_type = decisions[child_index * 7].type;

        if (early_leaf_node_exit) {
            decision_type = Decision::Type::LEAF;
        }

        switch (decision_type) {
        case Decision::Type::LEAF: {
            int triangle_count = count_primitives(child_index, nodes_bvh, indices_bvh);
            // Assume a single triangle to pass as a leaf node
            // /!\ Note that it invalidates any traversal that would need the correct primitive index
            // i.e. this is only suitable for the neural representation
            if (early_leaf_node_exit) {
                triangle_count = 1;
            }
            assert(triangle_count > 0 && triangle_count <= 3);

            // Three highest bits contain unary representation of triangle count
            for (int j = 0; j < triangle_count; j++) {
                node.meta[i] |= (1 << (j + 5));
            }

            node.meta[i] |= num_triangles;

            num_triangles += triangle_count;
            assert(num_triangles <= 24);

            if (leaf_nodes)
                leaf_nodes->push_back(nodes_bvh[child_index]);

            break;
        }
        case Decision::Type::INTERNAL: {
            assert(!early_leaf_node_exit);
            node.meta[i] = (i + 24) | 0b00100000;

            node.imask |= (1 << i);
            num_internal_nodes++;
            break;
        }
        default:
            assert(false);
        }
    }

    if (early_leaf_node_exit) {
        return;
    }

    for (int i = 0; i < num_internal_nodes; i++) {
        m_bvh8.m_nodes.emplace_back();
    }
    // NOTE: 'node' may have been invalidated by emplace_back on previous line
    node = m_bvh8.m_nodes[node_index_bvh8];

    assert(node.base_index_child + num_internal_nodes == m_bvh8.m_nodes.size());
    assert(node.base_index_triangle + num_triangles == m_bvh8.m_indices.size());

    // Recurse on Internal Nodes
    int offset = 0;
    for (int i = 0; i < 8; i++) {
        int child_index = children[i];
        if (child_index == INVALID_NODE)
            continue;

        if (node.imask & (1 << i)) {
            collapse(nodes_bvh, indices_bvh, node.base_index_child + offset++, child_index, depth + 1, leaf_nodes);
        }
    }
}

}