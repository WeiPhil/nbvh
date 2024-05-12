#pragma once

#include "glm_pch.h"

#include <vector>
namespace ntwr {

namespace neural {

    static const char *const lod_decision_types[] = {"EqualIterations", "EqualSplits"};
    enum LodDecisionType { EqualIterations, EqualSplits };

    struct BvhSplitScheduler {
    private:
        struct SchedulerTmpData {
            int current_num_splits         = 0;
            int current_splitting_interval = 50;
            int current_total_splits       = 0;
        };

        struct SchedulerStats {
            int total_splits           = 0;
            int num_cuts               = 0;
            int expected_bvh_num_nodes = 0;
            int expected_bvh_depth     = 0;
        };

    public:
        int m_splitting_interval       = 50;
        int m_last_split_iteration     = 2000;
        int m_start_split_num          = 1;
        float m_split_scaling          = 2.f;
        float m_split_interval_scaling = 2.f;
        SchedulerStats m_stats;

        LodDecisionType m_lod_decision_type = LodDecisionType::EqualIterations;

        void update_stats_and_reset(int bvh_start_num_nodes, std::vector<float> &splits_graph, int num_iterations)
        {
            m_data = SchedulerTmpData{m_start_split_num, m_splitting_interval, 0};

            m_stats.total_splits           = 0;
            m_stats.num_cuts               = 0;
            m_stats.expected_bvh_num_nodes = bvh_start_num_nodes;
            m_stats.expected_bvh_depth     = 0;

            splits_graph.resize(num_iterations);

            for (int i = 0; i < num_iterations; ++i) {
                int num_leaf_nodes = (m_stats.expected_bvh_num_nodes + 1) / 2;
                int splits         = get_bvh_splits(i);
                if (splits > 0) {
                    if (splits > num_leaf_nodes)
                        splits = num_leaf_nodes;
                    m_stats.total_splits += splits;
                    m_stats.expected_bvh_num_nodes += 2 * splits;
                    m_stats.num_cuts++;
                }
                splits_graph[i] = (float)m_stats.total_splits;
            }

            m_stats.expected_bvh_depth = int(log2(float(m_stats.expected_bvh_num_nodes)));

            // reset tmp data
            m_data = SchedulerTmpData{m_start_split_num, m_splitting_interval, 0};
        }

        int get_bvh_splits(int training_iteration)
        {
            if (training_iteration == 0) {
                m_data = SchedulerTmpData{m_start_split_num, m_splitting_interval, 0};
            }

            if (training_iteration > m_last_split_iteration) {
                return 0;
            }

            int num_splits = 0;
            if (training_iteration % m_splitting_interval == 0) {
                num_splits = m_data.current_num_splits;
                m_data.current_total_splits += num_splits;
            }

            if (training_iteration % m_data.current_splitting_interval == 0) {
                m_data.current_num_splits *= m_split_scaling;
                m_data.current_splitting_interval *= m_split_interval_scaling;
            }

            return num_splits;
        }

        bool should_assign_new_lod(int training_iteration,
                                   int current_learned_lod,
                                   int num_lods_learned,
                                   int max_bvh_lods)
        {
            bool assign_lod_and_decrement_lod = false;
            switch (m_lod_decision_type) {
            case LodDecisionType::EqualIterations: {
                int next_lod_assignment_iter =
                    (max_bvh_lods - current_learned_lod) * m_last_split_iteration / num_lods_learned;
                assign_lod_and_decrement_lod = next_lod_assignment_iter <= training_iteration;
                break;
            }
            case LodDecisionType::EqualSplits: {
                int next_lod_assignment_split_number =
                    (max_bvh_lods - current_learned_lod) * m_stats.total_splits / num_lods_learned;
                assign_lod_and_decrement_lod = next_lod_assignment_split_number <= m_data.current_total_splits;
                break;
            }
            default:
                UNREACHABLE();
                break;
            }
            assign_lod_and_decrement_lod |= training_iteration >= m_last_split_iteration;

            return assign_lod_and_decrement_lod;
        }

        inline nlohmann::json to_json()
        {
            return nlohmann::ordered_json{
                {"splitting_interval", m_splitting_interval},
                {"last_split_iteration", m_last_split_iteration},
                {"start_split_num", m_start_split_num},
                {"split_scaling", m_split_scaling},
                {"split_interval_scaling", m_split_interval_scaling},
            };
        }

        inline void from_json(nlohmann::json json_config)
        {
            BvhSplitScheduler default_config;
            m_splitting_interval   = json_config.value("splitting_interval", default_config.m_splitting_interval);
            m_last_split_iteration = json_config.value("last_split_iteration", default_config.m_last_split_iteration);
            m_start_split_num      = json_config.value("start_split_num", default_config.m_start_split_num);
            m_split_scaling        = json_config.value("split_scaling", default_config.m_split_scaling);
            m_split_interval_scaling =
                json_config.value("split_interval_scaling", default_config.m_split_interval_scaling);
        }

    private:
        SchedulerTmpData m_data;
    };

}
}