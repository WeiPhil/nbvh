#include "cuda/neural/neural_bvh_renderer.h"

#include <filesystem>
#include <tuple>

#include <bvh/v2/bvh.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/executor.h>
#include <bvh/v2/node.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/vec.h>

namespace ntwr {
namespace neural {

    // Make sure the index we use actually fit our neural bvh node!
    using LibNodeNeural   = bvh::v2::Node<float, 3, 64, 32>;
    using LibBVHNeural    = bvh::v2::Bvh<LibNodeNeural>;
    using LibBbbox        = bvh::v2::BBox<float, 3>;
    using LibVec3         = bvh::v2::Vec<float, 3>;
    using StdOutputStream = bvh::v2::StdOutputStream;

    // Traverse the BVH and build the neural bvh.
    void traverse_and_build_neural_bvh(const std::vector<BVH2Node> &bvh_nodes,
                                       uint32_t bvh_node_idx,
                                       uint32_t neural_node_index,
                                       std::vector<NeuralBVH2Node> &neural_nodes,
                                       uint32_t &neural_nodes_used,
                                       int depth,
                                       int max_depth,
                                       std::vector<uint32_t> &neural_node_map,
                                       std::vector<NeuralTreeCutData> &neural_tree_cut_data,
                                       int first_learned_lod)
    {
        const BVH2Node &node               = bvh_nodes[bvh_node_idx];
        neural_node_map[neural_node_index] = bvh_node_idx;
        // Copy node bounding box
        neural_nodes[neural_node_index].assign_bbox(node.bbox());
        neural_tree_cut_data[neural_node_index] = NeuralTreeCutData{
            0.f,              // loss
            1,                // iter_alive
            (uint32_t)depth,  // depth
            0,                // samples
        };

        if (node.is_leaf() || max_depth == depth) {
            // Set primitive number to pass as leaf node, we assume the lowest lod when building the tree
            neural_nodes[neural_node_index].prim_count = node.is_leaf() ? LEAF_FINAL : LEAF_LOD(first_learned_lod);
            neural_nodes[neural_node_index].left_or_prim_offset = NO_NEURAL_CHILD;
            return;
        }

        assert(!node.is_leaf());

        uint32_t left_node_idx  = neural_nodes_used++;
        uint32_t right_node_idx = neural_nodes_used++;

        // Space for the two childs
        neural_nodes.resize(neural_nodes_used);
        neural_node_map.resize(neural_nodes_used);
        neural_tree_cut_data.resize(neural_nodes_used);

        // This is an interior node, set pointer to left node
        neural_nodes[neural_node_index].prim_count          = 0;
        neural_nodes[neural_node_index].left_or_prim_offset = left_node_idx;

        traverse_and_build_neural_bvh(bvh_nodes,
                                      node.left_or_prim_offset,
                                      left_node_idx,
                                      neural_nodes,
                                      neural_nodes_used,
                                      depth + 1,
                                      max_depth,
                                      neural_node_map,
                                      neural_tree_cut_data,
                                      first_learned_lod);
        traverse_and_build_neural_bvh(bvh_nodes,
                                      node.left_or_prim_offset + 1,
                                      right_node_idx,
                                      neural_nodes,
                                      neural_nodes_used,
                                      depth + 1,
                                      max_depth,
                                      neural_node_map,
                                      neural_tree_cut_data,
                                      first_learned_lod);
    }

    // Traverse the Load BVH and build the neural bvh.
    void traverse_neural_bvh_and_update_states(const std::vector<BVH2Node> &bvh_nodes,
                                               const std::vector<NeuralBVH2Node> &neural_nodes,
                                               uint32_t bvh_node_idx,
                                               uint32_t neural_node_index,
                                               NeuralBvhStats &neural_bvh_stats,
                                               uint32_t &neural_nodes_used,
                                               int depth,
                                               uint32_t current_traversed_lod,
                                               std::vector<uint32_t> &neural_node_map,
                                               std::vector<NeuralTreeCutData> &neural_tree_cut_data,
                                               int &max_depth,
                                               uint32_t &min_traversed_lod)
    {
        const BVH2Node &node                    = bvh_nodes[bvh_node_idx];
        const NeuralBVH2Node &neural_node       = neural_nodes[neural_node_index];
        neural_node_map[neural_node_index]      = bvh_node_idx;
        neural_tree_cut_data[neural_node_index] = NeuralTreeCutData{
            0.f,              // loss
            1,                // iter_alive
            (uint32_t)depth,  // depth
            0,                // samples
        };

        if (!neural_node.has_neural_child()) {
            max_depth         = std::max(max_depth, depth);
            min_traversed_lod = std::min(min_traversed_lod, current_traversed_lod);

            for (int bvh_lod = current_traversed_lod; bvh_lod >= 0; bvh_lod--) {
                neural_bvh_stats.n_bvh_lod_nodes[bvh_lod]++;
            }
            return;
        }

        assert(!node.is_leaf());
        assert(neural_node.has_neural_child());

        // Decrement while leaf node as we might have lods sharing a leaf node
        for (int bvh_lod = current_traversed_lod; bvh_lod >= 0; bvh_lod--) {
            if (neural_node.is_leaf_at_lod(bvh_lod)) {
                neural_bvh_stats.n_bvh_lod_nodes[bvh_lod]++;
                current_traversed_lod--;
                assert(current_traversed_lod >= 0);
                assert(neural_node.is_interior_at_lod(bvh_lod) || neural_node.is_leaf_at_lod(bvh_lod));
            }
        }

        for (int bvh_lod = current_traversed_lod; bvh_lod >= 0; bvh_lod--) {
            if (neural_node.is_interior_at_lod(bvh_lod))
                neural_bvh_stats.n_bvh_lod_nodes[bvh_lod]++;
        }

        neural_nodes_used += 2;

        traverse_neural_bvh_and_update_states(bvh_nodes,
                                              neural_nodes,
                                              node.left_or_prim_offset,
                                              neural_node.left_offset(),
                                              neural_bvh_stats,
                                              neural_nodes_used,
                                              depth + 1,
                                              current_traversed_lod,
                                              neural_node_map,
                                              neural_tree_cut_data,
                                              max_depth,
                                              min_traversed_lod);
        traverse_neural_bvh_and_update_states(bvh_nodes,
                                              neural_nodes,
                                              node.left_or_prim_offset + 1,
                                              neural_node.left_offset() + 1,
                                              neural_bvh_stats,
                                              neural_nodes_used,
                                              depth + 1,
                                              current_traversed_lod,
                                              neural_node_map,
                                              neural_tree_cut_data,
                                              max_depth,
                                              min_traversed_lod);
    }

    void NeuralBVHRenderer::save_neural_bvh(const std::string &output_filename)
    {
        // First convert neural bvh back to serializable LibBVH

        // Download device data to host array
        std::vector<NeuralBVH2Node> neural_nodes;
        neural_nodes.resize(m_neural_nodes_used);
        assert(m_neural_nodes_used == m_neural_bvh.num_nodes);

        logger(
            LogLevel::Info, "Saving neural BVH with %d nodes to %s...", m_neural_nodes_used, output_filename.c_str());

        // Download from the device all the used nodes
        m_neural_bvh_nodes.download(neural_nodes.data(), m_neural_bvh.num_nodes, true);

        LibBVHNeural neural_libbvh;
        // Dummy prim_ids;
        neural_libbvh.prim_ids.resize(1);
        neural_libbvh.nodes.resize(m_neural_nodes_used);

        for (size_t node_idx = 0; node_idx < neural_libbvh.nodes.size(); node_idx++) {
            auto &libbvh_node       = neural_libbvh.nodes[node_idx];
            const auto &neural_node = neural_nodes[node_idx];
            const LibBbbox libbvh_bbox(LibVec3(neural_node.aabb_min.x, neural_node.aabb_min.y, neural_node.aabb_min.z),
                                       LibVec3(neural_node.aabb_max.x, neural_node.aabb_max.y, neural_node.aabb_max.z));
            libbvh_node.set_bbox(libbvh_bbox);
            libbvh_node.index.first_id   = neural_node.left_or_prim_offset;
            libbvh_node.index.prim_count = neural_node.prim_count;
        }

        std::ofstream out(output_filename, std::ofstream::binary);
        if (!out)
            logger(LogLevel::Error, "Failed writting BVH to file %s", output_filename.c_str());
        StdOutputStream stream(out);
        neural_libbvh.serialize(stream);
        logger(LogLevel::Info, "Done!");
    }

    void NeuralBVHRenderer::load_neural_bvh(const std::string &input_filename)
    {
        if (m_neural_bvh_nodes.d_ptr)
            m_neural_bvh_nodes.free();

        using StdInputStream = bvh::v2::StdInputStream;

        logger(LogLevel::Info, "Loading neural BVH from %s...", input_filename.c_str());

        std::ifstream in(input_filename, std::ofstream::binary);
        if (!in) {
            logger(LogLevel::Error, "Failed loading BVH from file %s", input_filename.c_str());
            return;
        }
        StdInputStream stream(in);
        LibBVHNeural neural_libbvh = LibBVHNeural::deserialize(stream);

        logger(LogLevel::Info, "Loaded neural LibBVH with %d nodes.", neural_libbvh.nodes.size());

        std::vector<NeuralBVH2Node> neural_nodes;

        neural_nodes.resize(neural_libbvh.nodes.size());

        for (size_t node_idx = 0; node_idx < neural_libbvh.nodes.size(); node_idx++) {
            auto &libbvh_node               = neural_libbvh.nodes[node_idx];
            const LibBbbox libbvh_bbox      = libbvh_node.get_bbox();
            neural_nodes[node_idx].aabb_min = glm::vec3(libbvh_bbox.min[0], libbvh_bbox.min[1], libbvh_bbox.min[2]);
            neural_nodes[node_idx].aabb_max = glm::vec3(libbvh_bbox.max[0], libbvh_bbox.max[1], libbvh_bbox.max[2]);
            neural_nodes[node_idx].left_or_prim_offset = libbvh_node.index.first_id;
            neural_nodes[node_idx].prim_count          = libbvh_node.index.prim_count;
        }
        m_neural_nodes_alloc_count = neural_nodes.size();
        m_neural_nodes_used        = m_neural_nodes_alloc_count;

        m_neural_bvh_nodes.alloc_and_upload(neural_nodes);
        m_neural_bvh.nodes     = m_neural_bvh_nodes.d_pointer();
        m_neural_bvh.num_nodes = m_neural_bvh_nodes.num_elements();
        assert(m_neural_nodes_used == m_neural_bvh.num_nodes);
        assert(m_neural_nodes_alloc_count == m_neural_bvh.num_nodes);

        BVH &neural_cpu_bvh = m_backend->m_cpu_mesh_bvhs[m_bvh_const.neural_bvh_idx];

        m_neural_scene_bbox =
            m_bvh_const.ellipsoidal_bvh ? neural_cpu_bvh.root_bbox().ellipsoid_bbox() : neural_cpu_bvh.root_bbox();

        reinitialise_neural_bvh_learning_states();
    }

    void NeuralBVHRenderer::split_neural_bvh_at(const std::vector<uint32_t> &split_neural_node_indices)
    {
        assert(m_neural_bvh_nodes.d_ptr);
        assert(m_neural_tree_cut_data.d_ptr);

        // Download device data to host array
        std::vector<NeuralBVH2Node> neural_nodes;
        neural_nodes.resize(m_neural_nodes_used);
        std::vector<NeuralTreeCutData> tree_cut_data;
        tree_cut_data.resize(m_neural_nodes_used);

        assert(m_neural_nodes_used == m_neural_bvh.num_nodes);

        // Get previous neural_node data (both cuda buffer might have more memory pre-allocated, allow downloading less)
        m_neural_bvh_nodes.download(neural_nodes.data(), m_neural_bvh.num_nodes, true);
        m_neural_tree_cut_data.download(tree_cut_data.data(), m_neural_bvh.num_nodes, true);

        // Make sure we don't split more nodes than possible at low depth
        size_t num_leaf_nodes     = (m_neural_bvh.num_nodes + 1) / 2;
        auto unique_split_indices = std::set<uint32_t>(
            split_neural_node_indices.begin(),
            split_neural_node_indices.begin() + min(num_leaf_nodes, split_neural_node_indices.size()));

        // Split original node

        auto &neural_cpu_bvh = m_backend->m_cpu_mesh_bvhs[m_bvh_const.neural_bvh_idx];

        for (uint32_t split_neural_node_idx : unique_split_indices) {
            // logger(LogLevel::Info, "Splitting neural bvh at node idx: %d", unique_split_indices.size());

            // Get corresponding original bvh node
            const BVH2Node &parent_node = neural_cpu_bvh.m_nodes[m_neural_node_map[split_neural_node_idx]];

            if (parent_node.is_leaf()) {
                logger(LogLevel::Warn, "Leaf node reached, no split");
                continue;
            }

            if (neural_nodes[split_neural_node_idx].has_neural_child()) {
                logger(LogLevel::Error, "Trying to split an interior node! Aborting.");
                continue;
            }

            // Not enough pre-allocated space on device ? resize
            if (m_neural_nodes_used + 2 > m_neural_nodes_alloc_count) {
                if (m_neural_bvh_nodes.d_ptr)
                    m_neural_bvh_nodes.free();
                if (m_neural_tree_cut_data.d_ptr)
                    m_neural_tree_cut_data.free();

                m_neural_nodes_alloc_count = max(4, m_neural_nodes_alloc_count * 2);
                m_neural_bvh_nodes.alloc(m_neural_nodes_alloc_count);
                m_neural_bvh_nodes.upload(neural_nodes.data(), m_neural_nodes_used);

                // Reupload tree cut data as well
                m_neural_tree_cut_data.alloc(m_neural_nodes_alloc_count);
                m_neural_tree_cut_data.upload(tree_cut_data.data(), m_neural_nodes_used);

                m_neural_bvh.nodes     = m_neural_bvh_nodes.d_pointer();
                m_neural_bvh.num_nodes = m_neural_nodes_used;
            }

            const BVH2Node &left_node  = neural_cpu_bvh.m_nodes[parent_node.left_or_prim_offset];
            const BVH2Node &right_node = neural_cpu_bvh.m_nodes[parent_node.left_or_prim_offset + 1];

            uint32_t left_neural_node_idx  = m_neural_nodes_used++;
            uint32_t right_neural_node_idx = m_neural_nodes_used++;

            // Update neural node map with newly added nodes
            m_neural_node_map.push_back(parent_node.left_or_prim_offset);
            m_neural_node_map.push_back(parent_node.left_or_prim_offset + 1);

            neural_nodes.resize(m_neural_nodes_used);
            tree_cut_data.resize(m_neural_nodes_used);

            assert(!neural_nodes[split_neural_node_idx].has_neural_child());

            // Make the leaf node an interior node if it is not already at the level
            if (neural_nodes[split_neural_node_idx].is_leaf_assigned_to_lod(m_bvh_const.current_learned_lod)) {
                neural_nodes[split_neural_node_idx].prim_count = 0;
            }
            neural_nodes[split_neural_node_idx].left_or_prim_offset = left_neural_node_idx;
            neural_nodes[split_neural_node_idx].assign_bbox(parent_node.bbox());
            assert(left_neural_node_idx != UINT32_MAX);
            assert(neural_nodes[split_neural_node_idx].has_neural_child());
            assert(neural_nodes[split_neural_node_idx].is_interior_at_lod(m_bvh_const.current_learned_lod));
            assert(!neural_nodes[split_neural_node_idx].is_leaf_at_lod(m_bvh_const.current_learned_lod));
            // Unchanged : tree_cut_data[split_neural_node_idx]

            // Then set primitive number to 1 to pass as leaf node
            // The primitive offset is used to indicate temporarilly if the neural node has a child for constructing all
            // lods. TODO : Later when considering the full hybrid neural bvh this should be overwritten with the real
            // primitive offset at final leaf nodes
            neural_nodes[left_neural_node_idx].prim_count =
                left_node.is_leaf() ? LEAF_FINAL : LEAF_LOD(m_bvh_const.current_learned_lod);
            neural_nodes[left_neural_node_idx].assign_bbox(left_node.bbox());
            neural_nodes[left_neural_node_idx].left_or_prim_offset = NO_NEURAL_CHILD;
            neural_nodes[right_neural_node_idx].prim_count =
                right_node.is_leaf() ? LEAF_FINAL : LEAF_LOD(m_bvh_const.current_learned_lod);
            neural_nodes[right_neural_node_idx].assign_bbox(right_node.bbox());
            neural_nodes[right_neural_node_idx].left_or_prim_offset = NO_NEURAL_CHILD;

            assert(neural_nodes[left_neural_node_idx].is_leaf_at_lod(m_bvh_const.current_learned_lod));
            assert(neural_nodes[right_neural_node_idx].is_leaf_at_lod(m_bvh_const.current_learned_lod));

            tree_cut_data[left_neural_node_idx] = NeuralTreeCutData{
                0.0f,                                            // loss
                1,                                               // iter_alive
                tree_cut_data[split_neural_node_idx].depth + 1,  // depth
                0,                                               // samples
            };
            tree_cut_data[right_neural_node_idx] = NeuralTreeCutData{
                0.0f,                                            // loss
                1,                                               // iter_alive
                tree_cut_data[split_neural_node_idx].depth + 1,  // depth
                0,                                               // samples
            };

            m_neural_tree_cut_max_depth =
                max(m_neural_tree_cut_max_depth, tree_cut_data[split_neural_node_idx].depth + 1);

            assert(m_neural_node_map.size() == neural_nodes.size());

            // And upload new data
            assert(m_neural_nodes_used <= m_neural_nodes_alloc_count);
            // needs two uploads since the data might not be contiguous
            m_neural_bvh_nodes.upload(&neural_nodes[split_neural_node_idx], 1, split_neural_node_idx);
            m_neural_bvh_nodes.upload(&neural_nodes[left_neural_node_idx], 2, left_neural_node_idx);

            m_neural_tree_cut_data.upload(&tree_cut_data[left_neural_node_idx], 2, left_neural_node_idx);

            // And update neural bvh node count
            m_neural_bvh.num_nodes = m_neural_nodes_used;
        }

        m_bvh_stats.n_bvh_lod_nodes[m_bvh_const.current_learned_lod] = m_neural_nodes_used;
    }

    void NeuralBVHRenderer::assign_lod_to_current_neural_leaf_nodes()
    {
        uint32_t assigned_lod_level = m_bvh_const.current_learned_lod;
        logger(LogLevel::Info, "Assigning BVH LoD Level %d", assigned_lod_level);

        assert(m_neural_bvh_nodes.d_ptr);
        assert(m_neural_tree_cut_data.d_ptr);

        // Download device data to host array
        std::vector<NeuralBVH2Node> neural_nodes;
        neural_nodes.resize(m_neural_nodes_used);

        assert(m_neural_bvh.num_nodes == m_neural_nodes_used);

        // Get previous neural_node data
        m_neural_bvh_nodes.download(neural_nodes.data(), m_neural_bvh.num_nodes, true);

        for (auto &neural_node : neural_nodes) {
            if (assigned_lod_level == (NUM_BVH_LODS - 1)) {
                assert((neural_node.has_neural_child() && neural_node.is_interior_at_lod(assigned_lod_level)) ||
                       (!neural_node.has_neural_child() && neural_node.is_leaf_at_lod(assigned_lod_level)));
                continue;
            }
            // Since we gradually build the lod from coarsest to finest lod and appended node always have the highest
            // lod level we can simply check if the node is a leaf at the highest bvh lod or if it has no neural child
            // (we would overwrite the previous leaf lod)
            if (!neural_node.has_neural_child()) {
                // We need both check just for the case assigned_lod_level is the maximum level
                assert(neural_node.is_leaf_at_lod(assigned_lod_level) ||
                       neural_node.is_leaf_at_lod(assigned_lod_level + 1));

                neural_node.prim_count = neural_node.is_final_leaf() ? LEAF_FINAL : LEAF_LOD(assigned_lod_level);
            }
        }

        // We set it again here as we might have lods with the same number of nodes which is not set during splitting
        m_bvh_stats.n_bvh_lod_nodes[assigned_lod_level] = m_neural_nodes_used;

        m_neural_bvh_nodes.upload(neural_nodes.data(), m_neural_nodes_used);

        m_all_bvh_lods_assigned = m_bvh_const.current_learned_lod == 0;
        if (!m_all_bvh_lods_assigned) {
            m_bvh_const.current_learned_lod--;
        } else {
            logger(LogLevel::Info, "All LoDs assigned!");
            m_tree_cut_learning = false;
        }
    }

    void NeuralBVHRenderer::reinitialise_neural_bvh_learning_states()
    {
        assert(m_neural_bvh_nodes.d_ptr);

        if (m_neural_tree_cut_data.d_ptr)
            m_neural_tree_cut_data.free();

        logger(LogLevel::Info, "Reinitialising neural bvh learning data..");

        std::vector<NeuralBVH2Node> neural_nodes;

        neural_nodes.resize(m_neural_nodes_used);

        assert(m_neural_nodes_used == m_neural_bvh.num_nodes);
        // Get previous neural_node data
        m_neural_bvh_nodes.download(neural_nodes.data(), m_neural_bvh.num_nodes, true);

        BVH &neural_cpu_bvh = m_backend->m_cpu_mesh_bvhs[m_bvh_const.neural_bvh_idx];

        m_bvh_stats.reset();
        m_bvh_stats.n_original_bvh_nodes = neural_cpu_bvh.m_nodes.size();

        std::vector<NeuralTreeCutData> neural_tree_cut_data;
        // Reset and make space for root node
        m_neural_node_map.resize(m_neural_nodes_used);
        neural_tree_cut_data.resize(m_neural_nodes_used);
        uint32_t neural_nodes_used     = 1;
        uint32_t current_traversed_lod = NUM_BVH_LODS - 1;
        uint32_t min_traversed_lod     = current_traversed_lod;
        int depth                      = 0;
        int max_depth                  = depth;

        traverse_neural_bvh_and_update_states(neural_cpu_bvh.m_nodes,
                                              neural_nodes,
                                              0,
                                              0,
                                              m_bvh_stats,
                                              neural_nodes_used,
                                              depth,
                                              current_traversed_lod,
                                              m_neural_node_map,
                                              neural_tree_cut_data,
                                              max_depth,
                                              min_traversed_lod);

        assert(neural_nodes_used == m_neural_nodes_used);
        assert(m_neural_node_map.size() == neural_nodes.size());
        assert(neural_tree_cut_data.size() == neural_nodes.size());

        m_neural_tree_cut_max_depth = max_depth;

        // Alloc as much as was allocated for the nodes
        m_neural_tree_cut_data.alloc(m_neural_nodes_alloc_count);
        m_neural_tree_cut_data.upload(neural_tree_cut_data.data(), neural_tree_cut_data.size());

        m_bvh_const.current_learned_lod =
            min_traversed_lod == (NUM_BVH_LODS - m_bvh_const.num_lods_learned) && m_bvh_const.num_lods_learned > 1
                ? 0
                : min_traversed_lod;

        m_all_bvh_lods_assigned = m_bvh_const.current_learned_lod == 0;

        logger(LogLevel::Info, "Done!");
    }

    void NeuralBVHRenderer::build_and_upload_neural_bvh()
    {
        if (m_neural_bvh_nodes.d_ptr)
            m_neural_bvh_nodes.free();
        if (m_neural_tree_cut_data.d_ptr)
            m_neural_tree_cut_data.free();

        logger(LogLevel::Info, "Building neural bvh with max depth %d ...", m_neural_bvh_min_depth);

        BVH &neural_cpu_bvh = m_backend->m_cpu_mesh_bvhs[m_bvh_const.neural_bvh_idx];

        m_bvh_const.current_learned_lod = NUM_BVH_LODS - 1;

        std::vector<NeuralTreeCutData> neural_tree_cut_data;
        std::vector<NeuralBVH2Node> neural_nodes;
        // Make space for root node
        neural_nodes.resize(1);
        m_neural_node_map.resize(1);
        neural_tree_cut_data.resize(1);
        uint32_t neural_nodes_used = 1;

        traverse_and_build_neural_bvh(neural_cpu_bvh.m_nodes,
                                      0,
                                      0,
                                      neural_nodes,
                                      neural_nodes_used,
                                      0,
                                      m_neural_bvh_min_depth,
                                      m_neural_node_map,
                                      neural_tree_cut_data,
                                      m_bvh_const.current_learned_lod);
        assert(m_neural_node_map.size() == neural_nodes.size());
        assert(neural_tree_cut_data.size() == neural_nodes.size());
        m_neural_bvh_nodes.alloc_and_upload(neural_nodes);
        m_neural_nodes_alloc_count  = neural_nodes.size();
        m_neural_nodes_used         = m_neural_nodes_alloc_count;
        m_neural_tree_cut_max_depth = m_neural_bvh_min_depth;
        m_neural_tree_cut_data.alloc_and_upload(neural_tree_cut_data);

        assert(m_neural_nodes_used == m_neural_nodes_alloc_count);

        m_neural_bvh.nodes     = m_neural_bvh_nodes.d_pointer();
        m_neural_bvh.num_nodes = m_neural_nodes_used;

        m_neural_scene_bbox =
            m_bvh_const.ellipsoidal_bvh ? neural_cpu_bvh.root_bbox().ellipsoid_bbox() : neural_cpu_bvh.root_bbox();

        logger(LogLevel::Info, "Done, total nodes : %d", m_neural_bvh_nodes.num_elements());
    }

}
}