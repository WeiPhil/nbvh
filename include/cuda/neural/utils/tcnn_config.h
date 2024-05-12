#pragma once

#include "utils/imgui_utils.h"

#include <json/json.hpp>
namespace ntwr {

inline int find_string_index(const char *const values[], const char *const target_string, int item_count)
{
    for (int i = 0; i < item_count; i++) {
        if (strcmp(target_string, values[i]) == 0)
            return i;
    }
    return -1;
}

static const int n_neurons_values[]            = {16, 32, 64, 128, 256, 512};
static const int n_hidden_layers_values[]      = {1, 2, 3, 4, 5, 6, 7, 8};
static const int n_features_per_level_values[] = {1, 2, 4, 8};

/******** Losses *******/

static const char *const loss_types[] = {
    "RelativeL2", "L2", "RelativeL1", "L1", "RelativeL2Luminance", "BinaryCrossEntropy", "LogL2", "LHalf"};

enum LossType : uint8_t {
    RelativeL2,
    L2,
    RelativeL1,
    L1,
    RelativeL2Luminance,
    BinaryCrossEntropy,
    LogL2,
    LHalf,
};

/******** Encodings *******/

static const char *const encoding_types[] = {
    "Identity", "OneBlob", "HashGrid", "TiledGrid", "DenseGrid", "Frequency", "TriangleWave", "SphericalHarmonics"};

enum EncodingType : uint8_t {
    Identity,
    OneBlob,
    HashGrid,
    TiledGrid,
    DenseGrid,
    Frequency,
    TriangleWave,
    SphericalHarmonics,
};

static const char *const reduction_types[] = {"Concatenation", "Sum", "Product"};

enum CompositeReductionType : uint8_t {
    Concatenation,
    Sum,
    Product,
};

static const char *const interpolation_types[] = {"Nearest", "Linear", "Smoothstep"};

enum InterpolationType : uint8_t { Nearest, Linear, Smoothstep, COUNT };

static const char *const hash_types[] = {"Prime", "CoherentPrime", "ReversedPrime", "Rng"};

enum HashType : uint8_t { Prime, CoherentPrime, ReversedPrime, Rng };

struct HashGridOptions {
    int n_levels                    = 8;
    int n_features_per_level        = 4;
    int log2_hashmap_size           = 18;
    int base_resolution             = 8;
    float per_level_scale           = 2.0f;
    InterpolationType interpolation = InterpolationType::Linear;
    bool stochastic_interpolation   = false;
    HashType hash                   = HashType::CoherentPrime;

    inline nlohmann::json to_json()
    {
        return nlohmann::ordered_json{
            {"n_levels", n_levels},
            {"n_features_per_level", n_features_per_level},
            {"log2_hashmap_size", log2_hashmap_size},
            {"base_resolution", base_resolution},
            {"per_level_scale", per_level_scale},
            {"interpolation", interpolation_types[interpolation]},
            {"stochastic_interpolation", stochastic_interpolation},
            {"hash", hash_types[hash]},
        };
    }

    inline void from_json(nlohmann::json json_config)
    {
        HashGridOptions default_config;
        n_levels             = json_config.value("n_levels", default_config.n_levels);
        n_features_per_level = json_config.value("n_features_per_level", default_config.n_features_per_level);
        log2_hashmap_size    = json_config.value("log2_hashmap_size", default_config.log2_hashmap_size);
        base_resolution      = json_config.value("base_resolution", default_config.base_resolution);
        per_level_scale      = json_config.value("per_level_scale", default_config.per_level_scale);
        interpolation        = (InterpolationType)imgui_utils::find_index(
            json_config.value("interpolation", interpolation_types[default_config.interpolation]).c_str(),
            interpolation_types,
            ARRAY_SIZE(interpolation_types));
        stochastic_interpolation =
            json_config.value("stochastic_interpolation", default_config.stochastic_interpolation);
        hash = (HashType)imgui_utils::find_index(
            json_config.value("hash", hash_types[default_config.hash]).c_str(), hash_types, ARRAY_SIZE(hash_types));
    }
};

inline bool hashgrid_encoding_ui(HashGridOptions &hashgrid_options, std::string component_id)
{
    bool hashgrid_changed = false;

    hashgrid_changed |= ImGui::InputInt(("n_levels##" + component_id).c_str(), &hashgrid_options.n_levels);

    hashgrid_changed |= imgui_utils::combo_box_int(("n_features_per_level##" + component_id).c_str(),
                                                   n_features_per_level_values,
                                                   ARRAY_SIZE(n_features_per_level_values),
                                                   hashgrid_options.n_features_per_level);
    hashgrid_changed |=
        ImGui::InputInt(("log2_hashmap_size##" + component_id).c_str(), &hashgrid_options.log2_hashmap_size);

    hashgrid_changed |=
        ImGui::InputInt(("base_resolution##" + component_id).c_str(), &hashgrid_options.base_resolution);
    hashgrid_changed |=
        ImGui::InputFloat(("per_level_scale##" + component_id).c_str(), &hashgrid_options.per_level_scale);

    hashgrid_changed |= ImGui::Checkbox(("stochastic_interpolation##" + component_id).c_str(),
                                        &hashgrid_options.stochastic_interpolation);
    hashgrid_changed |= imgui_utils::combo_box(("interpolation##" + component_id).c_str(),
                                               interpolation_types,
                                               ARRAY_SIZE(interpolation_types),
                                               hashgrid_options.interpolation);

    hashgrid_changed |= imgui_utils::combo_box(
        ("Hashing##" + component_id).c_str(), hash_types, ARRAY_SIZE(hash_types), hashgrid_options.hash);

    return hashgrid_changed;
}

struct ShOptions {
    int max_degree = 8;
};

/******** Optimizers *******/

static const char *const optimizer_types[] = {"SGD", "Adam", "Shampoo", "Novograd"};

enum OptimizerType : uint8_t { SGD, Adam, Shampoo, Novograd };

struct OptimizerOptions {
    OptimizerType type  = OptimizerType::Adam;
    float learning_rate = 1e-2f;
    int decay_start     = 10000;
    int decay_interval  = 2000;
    float decay_base    = 0.33f;
    float beta1         = 0.9f;    // Beta1 parameter of Adam.
    float beta2         = 0.999f;  // Beta2 parameter of Adam.
    float epsilon       = 1e-8f;   // Epsilon parameter of Adam.
    float l2_reg        = 1e-8f;   // Strength of L2 regularization
                                   // applied to the to-be-optimized params.
    float relative_decay = 0.0f;   // Percentage of weights lost per step.
    float absolute_decay = 0.0f;   // Amount of weights lost per step.
    bool adabound        = false;  // Whether to enable AdaBound.

    inline nlohmann::json to_json()
    {
        return nlohmann::ordered_json{
            {"type", optimizer_types[type]},
            {"learning_rate", learning_rate},
            {"decay_start", decay_start},
            {"decay_interval", decay_interval},
            {"decay_base", decay_base},
            {"beta1", beta1},
            {"beta2", beta2},
            {"epsilon", epsilon},
            {"l2_reg", l2_reg},
            {"relative_decay", relative_decay},
            {"absolute_decay", absolute_decay},
            {"adabound", adabound},
        };
    }

    inline void from_json(nlohmann::json json_config)
    {
        OptimizerOptions default_config;
        type = (OptimizerType)imgui_utils::find_index(
            json_config.value("type", optimizer_types[default_config.type]).c_str(),
            optimizer_types,
            ARRAY_SIZE(optimizer_types));
        learning_rate  = json_config.value("learning_rate", default_config.learning_rate);
        decay_start    = json_config.value("decay_start", default_config.decay_start);
        decay_interval = json_config.value("decay_interval", default_config.decay_interval);
        decay_base     = json_config.value("decay_base", default_config.decay_base);
        beta1          = json_config.value("beta1", default_config.beta1);
        beta2          = json_config.value("beta2", default_config.beta2);
        epsilon        = json_config.value("epsilon", default_config.epsilon);
        l2_reg         = json_config.value("l2_reg", default_config.l2_reg);
        relative_decay = json_config.value("relative_decay", default_config.relative_decay);
        absolute_decay = json_config.value("absolute_decay", default_config.absolute_decay);
        adabound       = json_config.value("adabound", default_config.adabound);
    }
};

/******** Networks *******/

static const char *const network_types[] = {"FullyFusedMLP", "CutlassMLP"};

enum NetworkType : uint8_t { FullyFusedMLP, CutlassMLP };

/******** Activations *******/

static const char *const activation_types[] = {
    "ReLU", "Exponential", "Sigmoid", "Squareplus", "Softplus", "Tanh", "None"};

enum ActivationType : uint8_t {
    ReLU,
    Exponential,
    Sigmoid,
    Squareplus,
    Softplus,
    Tanh,
    None,
};

struct NetworkOptions {
    NetworkType network              = NetworkType::FullyFusedMLP;
    ActivationType activation        = ActivationType::ReLU;
    ActivationType output_activation = ActivationType::Sigmoid;
    int n_neurons                    = 64;
    int n_hidden_layers              = 4;

    inline nlohmann::json to_json()
    {
        return nlohmann::ordered_json{
            {"network", network_types[network]},
            {"activation", activation_types[activation]},
            {"output_activation", activation_types[output_activation]},
            {"n_neurons", n_neurons},
            {"n_hidden_layers", n_hidden_layers},
        };
    }

    inline void from_json(nlohmann::json json_config)
    {
        NetworkOptions default_config;
        network = (NetworkType)imgui_utils::find_index(
            json_config.value("network", network_types[default_config.network]).c_str(),
            network_types,
            ARRAY_SIZE(network_types));
        activation = (ActivationType)imgui_utils::find_index(
            json_config.value("activation", activation_types[default_config.activation]).c_str(),
            activation_types,
            ARRAY_SIZE(activation_types));
        output_activation = (ActivationType)imgui_utils::find_index(
            json_config.value("output_activation", activation_types[default_config.output_activation]).c_str(),
            activation_types,
            ARRAY_SIZE(activation_types));
        n_neurons       = json_config.value("n_neurons", default_config.n_neurons);
        n_hidden_layers = json_config.value("n_hidden_layers", default_config.n_hidden_layers);
    }
};

}