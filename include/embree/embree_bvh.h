#pragma once

#include <embree4/rtcore.h>

#include <atomic>
#include <vector>

#include "base/bbox.h"
#include "utils/logger.h"

namespace ntwr {

struct EmbreeNode {
    Bbox3f bbox;
    bool is_leaf_node;
};

struct EmbreeBuildInfo {
    EmbreeBuildInfo() : n_nodes(0), n_interior_nodes(0), n_leaf_nodes(0), n_primitives(0) {}

    // Explicitly define a copy constructor
    EmbreeBuildInfo(const EmbreeBuildInfo &other)
        : n_nodes(other.n_nodes.load()),
          n_interior_nodes(other.n_interior_nodes.load()),
          n_leaf_nodes(other.n_leaf_nodes.load()),
          n_primitives(other.n_primitives.load())
    {
    }
    std::atomic<uint32_t> n_nodes;
    std::atomic<uint32_t> n_interior_nodes;
    std::atomic<uint32_t> n_leaf_nodes;
    std::atomic<uint32_t> n_primitives;
};

struct EmbreeInteriorNode : public EmbreeNode {
    EmbreeNode *children[2];

    static void *create(RTCThreadLocalAllocator alloc, unsigned int num_children, void *user_ptr)
    {
        assert(num_children == 2);
        // void *ptr = rtcThreadLocalAlloc(alloc, sizeof(InnerNode), 16);
        void *ptr = rtcThreadLocalAlloc(alloc, sizeof(EmbreeInteriorNode), 16);

        EmbreeBuildInfo *build_info = static_cast<EmbreeBuildInfo *>(user_ptr);
        build_info->n_nodes++;
        build_info->n_interior_nodes++;

        EmbreeInteriorNode *inner_node = static_cast<EmbreeInteriorNode *>(ptr);
        inner_node->is_leaf_node       = false;
        inner_node->bbox               = {glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX)};
        inner_node->children[0]        = nullptr;
        inner_node->children[1]        = nullptr;

        return (void *)inner_node;
    }

    static void set_children(void *node_ptr, void **child_ptr, unsigned int num_children, void *user_ptr)
    {
        assert(node_ptr);
        assert(num_children == 2);

        for (size_t i = 0; i < 2; i++) {
            assert(child_ptr[i]);
            EmbreeInteriorNode *inner_node = static_cast<EmbreeInteriorNode *>(node_ptr);
            inner_node->children[i]        = static_cast<EmbreeNode *>(child_ptr[i]);
        }
    }

    static void set_bounds(void *node_ptr, const RTCBounds **bounds, unsigned int num_children, void *user_ptr)
    {
        assert(num_children == 2);
        Bbox3f node_bbox = {glm::vec3(FLT_MAX), glm::vec3(-FLT_MAX)};
        for (size_t i = 0; i < 2; i++) {
            Bbox3f child_bbox = {glm::vec3(bounds[i]->lower_x, bounds[i]->lower_y, bounds[i]->lower_z),
                                 glm::vec3(bounds[i]->upper_x, bounds[i]->upper_y, bounds[i]->upper_z)};
            // logger(LogLevel::Info,
            //         "child_bbox[%d] : min(%f,%f,%f) max(%f,%f,%f)",
            //         i,
            //         bounds[i]->lower_x,
            //         bounds[i]->lower_y,
            //         bounds[i]->lower_z,
            //         bounds[i]->upper_x,
            //         bounds[i]->upper_y,
            //         bounds[i]->upper_z);
            node_bbox.expand(child_bbox);
        }
        // logger(LogLevel::Info,
        //         "parent : min(%f,%f,%f) max(%f,%f,%f)",
        //         node_bbox.min.x,
        //         node_bbox.min.y,
        //         node_bbox.min.z,
        //         node_bbox.max.x,
        //         node_bbox.max.y,
        //         node_bbox.max.z);
        EmbreeNode *node = static_cast<EmbreeNode *>(node_ptr);
        node->bbox       = node_bbox;
    }
};

struct EmbreeLeafNode : public EmbreeNode {
    uint32_t prim_id;
    uint32_t prim_count;

    static void *create(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive *prims, size_t num_prims, void *user_ptr)
    {
        assert(num_prims == 1);
        void *ptr = rtcThreadLocalAlloc(alloc, sizeof(EmbreeLeafNode), 16);

        EmbreeBuildInfo *build_info = static_cast<EmbreeBuildInfo *>(user_ptr);
        build_info->n_nodes++;
        build_info->n_leaf_nodes++;
        build_info->n_primitives += num_prims;

        ntwr::Bbox3f prim_bound = ntwr::Bbox3f{glm::vec3(prims->lower_x, prims->lower_y, prims->lower_z),
                                               glm::vec3(prims->upper_x, prims->upper_y, prims->upper_z)};

        EmbreeLeafNode *leaf_node = static_cast<EmbreeLeafNode *>(ptr);
        leaf_node->is_leaf_node   = true;
        leaf_node->bbox           = prim_bound;
        leaf_node->prim_id        = (uint32_t)prims->primID;
        leaf_node->prim_count     = (uint32_t)num_prims;

        return (void *)leaf_node;
    }
};

struct EmbreeBVH {
public:
    EmbreeBVH();
    ~EmbreeBVH();

    void build(std::vector<RTCBuildPrimitive> &prims_i);
    EmbreeNode *get_root() const;
    EmbreeBuildInfo get_build_info() const
    {
        return m_build_info;
    }
    bool ready() const
    {
        return m_ready;
    }

private:
    RTCDevice m_device;
    RTCBVH m_bvh;
    bool m_ready       = false;
    EmbreeNode *m_root = nullptr;
    EmbreeBuildInfo m_build_info;
};

}