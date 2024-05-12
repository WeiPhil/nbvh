

#include "base/accel/bvh.h"
#include "base/accel/bvh8_converter.h"
#include "cuda/cuda_backend.h"
#include "utils/logger.h"

#include <iostream>
namespace ntwr {

// Only build with C++ compiler (cuda C++ 20 features not supported)
#ifndef __CUDACC__
using LibNode = bvh::v2::Node<float, 3>;
using LibBVH  = bvh::v2::Bvh<LibNode>;
#endif

void BVH2Node::assign_bbox(Bbox3f bbox)
{
    aabb_min = bbox.min;
    aabb_max = bbox.max;
}

void BVH2Node::print_node()
{
    if (this->is_leaf()) {
        logger(LogLevel::Info,
               "BVH2Node[leaf, primitive_offset=%d\n, bbox = [min(%f,%f,%f) "
               "max(%f,%f,%f)]\n, primitive_count=%d\n]",
               this->primitive_offset(),
               this->bbox().min.x,
               this->bbox().min.y,
               this->bbox().min.z,
               this->bbox().max.x,
               this->bbox().max.y,
               this->bbox().max.z,
               this->primitive_count());

    } else {
        logger(LogLevel::Info,
               "BVH2Node[interior bbox = [min(%f,%f,%f)max(%f,%f,%f)]\n "
               ", left_offset=%d\n]",
               this->bbox().min.x,
               this->bbox().min.y,
               this->bbox().min.z,
               this->bbox().max.x,
               this->bbox().max.y,
               this->bbox().max.z,
               this->left_offset());
    }
}

// Traverse the BVH and write bounding boxes to the OBJ file.
void BVH2::write_BVH_to_OBJ(ObjWriter &obj_writer, uint32_t node_idx, int depth, int current_level, int max_depth_level)
{
    BVH2Node &node = m_nodes[node_idx];

    // Check if we have descended to a new level in the BVH.
    if (depth > current_level || node_idx == 0) {
        current_level = depth;
        // Assign a unique group name for this level.
        obj_writer.add_group("o bvh_level_" + std::to_string(current_level));
    }

    if (max_depth_level == -1 || depth <= max_depth_level) {
        obj_writer.add_bbox(node.bbox());
    }

    if (max_depth_level == depth)
        return;
    // Recursively traverse child nodes.
    if (node.is_leaf()) {
        return;  // Stop if it's a leaf node.
    } else {
        write_BVH_to_OBJ(obj_writer, node.left_or_prim_offset, depth + 1, current_level, max_depth_level);
        write_BVH_to_OBJ(obj_writer, node.left_or_prim_offset + 1, depth + 1, current_level, max_depth_level);
    }
}

void BVH2::generate_node_graph(uint32_t node_idx, std::ofstream &dot_file)
{
    BVH2Node &node = m_nodes[node_idx];

    std::string shape_string = node.is_leaf() ? "square" : "circle";

    // Shape and node id
    dot_file << "  " << node_idx << " [shape=" << shape_string << " label=\"Node " << node_idx;

    bool write_bbox = false;
    if (write_bbox) {
        dot_file << "\\nAABB: (" << node.aabb_min.x << ", " << node.aabb_min.y << ", " << node.aabb_min.z << ") - ("
                 << node.aabb_max.x << ", " << node.aabb_max.y << ", " << node.aabb_max.z << ")";
    }
    bool write_left = true;
    if (write_left) {
        dot_file << "\\nleft_or_prim_offset:" << node.left_or_prim_offset << "\\nprim_count:" << node.prim_count;
    }
    dot_file << "\"]" << std::endl;

    if (node.is_leaf()) {
        return;  // Stop if it's a leaf node.
    } else {
        dot_file << "  " << node_idx << "-> {" << node.left_or_prim_offset << "," << node.left_or_prim_offset + 1 << "}"
                 << std::endl;
        generate_node_graph(node.left_or_prim_offset, dot_file);
        generate_node_graph(node.left_or_prim_offset + 1, dot_file);
    }
}

void BVH2::write_BVH_graph()
{
    // Open a DOT file for writing
    std::ofstream dot_file("bvh2_graph.dot");
    if (!dot_file.is_open()) {
        logger(LogLevel::Error, "Error: Could not open the DOT file for writing.");
        exit(1);
    }
    // Write DOT header
    dot_file << "digraph BVH2 {" << std::endl;

    // Generate the DOT representation of the BVH
    generate_node_graph(0, dot_file);

    // Write DOT footer
    dot_file << "}" << std::endl;

    // Close the DOT file
    dot_file.close();
}

// Only build with C++ compiler (cuda C++20 features not supported)
#ifndef __CUDACC__

void BVH2::build_from_libbvh(const LibBVH &lib_bvh)
{
    m_nodes.resize(lib_bvh.nodes.size());

    // logger(LogLevel::Info, "BVH contains %d nodes", lib_bvh.nodes.size());

    for (size_t node_id = 0; node_id < lib_bvh.nodes.size(); node_id++) {
        /* code */
        auto node                 = lib_bvh.nodes[node_id];
        auto bbox                 = lib_bvh.nodes[node_id].get_bbox();
        m_nodes[node_id].aabb_min = glm::vec3(bbox.min[0], bbox.min[1], bbox.min[2]);
        m_nodes[node_id].aabb_max = glm::vec3(bbox.max[0], bbox.max[1], bbox.max[2]);

        assert(node.is_leaf() || (!node.is_leaf() && node.index.prim_count == 0));

        m_nodes[node_id].prim_count          = node.index.prim_count;
        m_nodes[node_id].left_or_prim_offset = node.index.first_id;
    }
    m_indices = lib_bvh.prim_ids;  // copy

    // logger(LogLevel::Info, "Writting bvh graphviz...");

    // write_BVH_graph();

#if WRITE_BVH_OBJ
    logger(LogLevel::Info, "Writting bvh nodes to obj...");

    ObjWriter m_obj_writer = ObjWriter("bvh_boxes.obj");

    // Initialize a variable to keep track of the current BVH level.
    int current_level = 0;

    write_BVH_to_OBJ(m_obj_writer, 0, 0, current_level, 5);

    m_obj_writer.close();
    logger(LogLevel::Info, "Done!");
#endif
}

// Dummy constructors required to allow the unique_ptr of incomplete type (for us the BVH8Converter)
BVH8::BVH8() {}
BVH8::~BVH8() {}

const BVH8Converter &BVH8::bvh_converter()
{
    return *m_bvh_converter;
}

void BVH8::build_from_libbvh(const LibBVH &lib_bvh)
{
    logger(LogLevel::Info, "LibBVH contains %d nodes before conversion", lib_bvh.nodes.size());

    BVH2 tmp_bvh2;
    tmp_bvh2.build_from_libbvh(lib_bvh);

    m_root_bbox = tmp_bvh2.root_bbox();

    m_bvh_converter = std::make_unique<BVH8Converter>(BVH8Converter(*this, tmp_bvh2));
    m_bvh_converter->convert();

    logger(LogLevel::Info, "BVH8 contains %d nodes after conversion", m_nodes.size());

#if WRITE_BVH_GRAPH
    logger(LogLevel::Info, "Writting bvh graphviz...");

    write_BVH_graph();
#endif
}

void BVH8::build_fixed_depth_from_cpu_bvh(const BVH2 &bvh,
                                          int max_depth,
                                          const BVH8Converter &precomputed_bvh8_converter,
                                          std::vector<BVH2Node> *leaf_nodes)
{
    logger(LogLevel::Info, "BVH contains %d nodes before conversion", bvh.m_nodes.size());

    m_bvh_converter = std::make_unique<BVH8Converter>(BVH8Converter(*this, bvh, max_depth, precomputed_bvh8_converter));
    m_bvh_converter->convert(leaf_nodes);

    logger(LogLevel::Info, "Fixed depth BVH8 contains %d nodes after conversion", m_nodes.size());

#if WRITE_BVH_GRAPH
    logger(LogLevel::Info, "Writting bvh graphviz...");

    write_BVH_graph();
#endif
}

void BVH8::generate_node_graph(uint32_t node_idx, std::ofstream &dot_file)
{
    BVH8Node &node = m_nodes[node_idx];

    assert(node.get_child_count() > 0);

    std::string node_id = "I" + std::to_string(node_idx);
    // Interior node definition
    dot_file << "  " << node_id << " [shape=circle"
             << " label=\"" << node_id;
    dot_file << "\"]" << std::endl;

    // Graph hierarchy (assuming this current node is an interior node)
    std::string parenting = "  " + node_id + "-> {";
    int offset            = 0;
    for (size_t child_idx = 0; child_idx < 8; child_idx++) {
        if (node.is_empty(child_idx))
            continue;

        if (!node.is_leaf(child_idx)) {
            int child_node_idx        = node.base_index_child + offset++;
            std::string child_node_id = "I" + std::to_string(child_node_idx);
            parenting += child_node_id;
        } else {
            std::string triangle_node_id = "L" + std::to_string(node.get_triangle_index(child_idx));
            parenting += triangle_node_id;
        }
        parenting += ",";
    }
    // dropping the last comma
    parenting.pop_back();
    dot_file << parenting << "}" << std::endl;

    offset = 0;
    for (int child_idx = 0; child_idx < 8; child_idx++) {
        if (node.is_empty(child_idx))
            continue;

        if (!node.is_leaf(child_idx)) {
            // Empty nodes or Leaf nodes need to be skipped to count the offset to the next node
            // e.g. if the child_idx is 4 and base index is 0 but child 0,1 and 2 are leaf nodes then the child node
            // index is just 1
            generate_node_graph(node.base_index_child + offset++, dot_file);
        } else {
            uint32_t triangle_idx = node.get_triangle_index(child_idx);
            assert(triangle_idx < m_indices.size());
            std::string child_node_id = "L" + std::to_string(triangle_idx);
            // Leaf Node definition
            dot_file << "  " << child_node_id
                     << " [shape=square"
                     //  << " label=\"" << child_node_id << " tri idx:" << (node.meta[child_idx] & 0b00011111);
                     << " label=\"" << child_node_id << " tri idx:" << node.get_triangle_index(child_idx)
                     << " num triangles" << node.get_num_triangles(child_idx);
            dot_file << "\"]" << std::endl;
        }
    }
}

void BVH8::write_BVH_graph()
{
    // Open a DOT file for writing
    std::ofstream dot_file("bvh8_graph.dot");
    if (!dot_file.is_open()) {
        logger(LogLevel::Error, "Error: Could not open the DOT file for writing.");
        exit(1);
    }

    uint32_t leaf_count     = 0;
    uint32_t interior_count = 0;
    for (size_t i = 0; i < m_nodes.size(); i++) {
        interior_count++;
        for (size_t child_idx = 0; child_idx < 8; child_idx++) {
            if (m_nodes[i].is_leaf(child_idx) && !m_nodes[i].is_empty(child_idx)) {
                leaf_count++;
            }
        }
    }
    logger(LogLevel::Info, "interior count %d leaf count %d", interior_count, leaf_count);

    // Write DOT header
    dot_file << "digraph BVH8 {" << std::endl;

    // Generate the DOT representation of the BVH
    generate_node_graph(0, dot_file);

    // Write DOT footer
    dot_file << "}" << std::endl;

    // Close the DOT file
    dot_file.close();
}

#endif

}