#pragma once

#include "base/bbox.h"
#include <json/json.hpp>

#ifndef NEURAL_BVH_CONSTANT
#define NEURAL_BVH_CONSTANT extern __constant__
#endif

namespace ntwr {
namespace neural {

    struct alignas(8) NeuralBVHRendererConstants {
        uint32_t neural_bvh_idx = 0;  // Non-configurable

        int num_lods_learned    = 1;
        int current_learned_lod = 0;
        int rendered_lod        = 0;
        bool ignore_neural_blas = false;
        bool path_lod_switch    = false;
        int lod_switch          = 0;

        float leaf_boundary_sampling_ratio = 0.f;

        float inference_mint_epsilon = 1e-4f;
        float inference_maxt_epsilon = 1e-3f;
        float learning_mint_epsilon  = 1e-3f;
        float learning_maxt_epsilon  = 1e-3f;
        float sampling_bbox_scaling  = 1.5f;
        float fixed_node_dilation    = 1e-5f;
        float node_extent_dilation   = 1e-4f;
        float loss_exposure          = 2e-7f;
        bool show_loss_heuristic     = false;

        bool ellipsoidal_bvh   = false;
        bool rescale_ellipsoid = false;

        int log2_bvh_max_traversals      = 0;
        bool sample_cosine_weighted      = true;
        bool backface_bounce_training    = false;
        int max_training_bounces         = 1;
        bool training_node_survival      = true;
        float survival_probability_scale = 1.f;
        float min_rr_survival_prob       = 0.005f;

        bool view_dependent_sampling = false;
        float view_dependent_ratio   = 0.f;

        inline NTWR_HOST nlohmann::json to_json()
        {
            return nlohmann::ordered_json{
                {"num_lods_learned", num_lods_learned},
                {"current_learned_lod", current_learned_lod},
                {"rendered_lod", rendered_lod},
                {"ignore_neural_blas", ignore_neural_blas},
                {"leaf_boundary_sampling_ratio", leaf_boundary_sampling_ratio},
                {"inference_mint_epsilon", inference_mint_epsilon},
                {"inference_maxt_epsilon", inference_maxt_epsilon},
                {"learning_mint_epsilon", learning_mint_epsilon},
                {"learning_maxt_epsilon", learning_maxt_epsilon},
                {"sampling_bbox_scaling", sampling_bbox_scaling},
                {"fixed_node_dilation", fixed_node_dilation},
                {"node_extent_dilation", node_extent_dilation},
                {"loss_exposure", loss_exposure},
                {"show_loss_heuristic", show_loss_heuristic},
                {"ellipsoidal_bvh", ellipsoidal_bvh},
                {"rescale_ellipsoid", rescale_ellipsoid},
                {"log2_bvh_max_traversals", log2_bvh_max_traversals},
                {"sample_cosine_weighted", sample_cosine_weighted},
                {"backface_bounce_training", backface_bounce_training},
                {"max_training_bounces", max_training_bounces},
                {"training_node_survival", training_node_survival},
                {"survival_probability_scale", survival_probability_scale},
                {"min_rr_survival_prob", min_rr_survival_prob},
            };
        }

        inline NTWR_HOST void from_json(nlohmann::json json_config)
        {
            NeuralBVHRendererConstants default_config;
            num_lods_learned    = json_config.value("num_lods_learned", default_config.num_lods_learned);
            current_learned_lod = json_config.value("current_learned_lod", default_config.current_learned_lod);
            rendered_lod        = json_config.value("rendered_lod", default_config.rendered_lod);
            ignore_neural_blas  = json_config.value("ignore_neural_blas", default_config.ignore_neural_blas);
            leaf_boundary_sampling_ratio =
                json_config.value("leaf_boundary_sampling_ratio", default_config.leaf_boundary_sampling_ratio);
            inference_mint_epsilon = json_config.value("inference_mint_epsilon", default_config.inference_mint_epsilon);
            inference_maxt_epsilon = json_config.value("inference_maxt_epsilon", default_config.inference_maxt_epsilon);
            learning_mint_epsilon  = json_config.value("learning_mint_epsilon", default_config.learning_mint_epsilon);
            learning_maxt_epsilon  = json_config.value("learning_maxt_epsilon", default_config.learning_maxt_epsilon);
            sampling_bbox_scaling  = json_config.value("sampling_bbox_scaling", default_config.sampling_bbox_scaling);
            fixed_node_dilation    = json_config.value("fixed_node_dilation", default_config.fixed_node_dilation);
            node_extent_dilation   = json_config.value("node_extent_dilation", default_config.node_extent_dilation);
            loss_exposure          = json_config.value("loss_exposure", default_config.loss_exposure);
            show_loss_heuristic    = json_config.value("show_loss_heuristic", default_config.show_loss_heuristic);
            ellipsoidal_bvh        = json_config.value("ellipsoidal_bvh", default_config.ellipsoidal_bvh);
            rescale_ellipsoid      = json_config.value("rescale_ellipsoid", default_config.rescale_ellipsoid);
            log2_bvh_max_traversals =
                json_config.value("log2_bvh_max_traversals", default_config.log2_bvh_max_traversals);
            sample_cosine_weighted = json_config.value("sample_cosine_weighted", default_config.sample_cosine_weighted);
            backface_bounce_training =
                json_config.value("backface_bounce_training", default_config.backface_bounce_training);
            max_training_bounces   = json_config.value("max_training_bounces", default_config.max_training_bounces);
            training_node_survival = json_config.value("training_node_survival", default_config.training_node_survival);
            survival_probability_scale =
                json_config.value("survival_probability_scale", default_config.survival_probability_scale);
            min_rr_survival_prob = json_config.value("min_rr_survival_prob", default_config.min_rr_survival_prob);
        }
    };

    NEURAL_BVH_CONSTANT NeuralBVHRendererConstants bvh_const;

    NEURAL_BVH_CONSTANT Bbox3f neural_scene_bbox;
}
}

#undef NEURAL_BVH_CONSTANT
