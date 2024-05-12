#pragma once

#include "imgui.h"

#include <string>
#include <vector>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))
#endif

namespace ntwr {

namespace imgui_utils {

    inline void large_separator()
    {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    inline int find_index(const int value, const int values[], int item_count)
    {
        for (int i = 0; i < item_count; i++) {
            if (values[i] == value)
                return i;
        }
        return -1;
    }

    inline int find_index(const char *value, const char *const values[], int item_count)
    {
        for (int i = 0; i < item_count; i++) {
            if (strcmp(values[i], value) == 0)
                return i;
        }
        return -1;
    }

    template <typename T>
    inline bool combo_box_int(const char *name, const int index_items[], int item_count, T &selected_value)
    {
        std::vector<std::string> str_items;
        for (int k = 0; k < item_count; k++)
            str_items.push_back(std::to_string(index_items[k]));

        int selected_idx = find_index((int)selected_value, index_items, item_count);
        bool changed     = false;
        if (ImGui::BeginCombo(name, str_items[selected_idx].c_str())) {
            for (int i = 0; i < item_count; ++i) {
                bool is_selected = (i == selected_idx);
                if (ImGui::Selectable(str_items[i].c_str(), is_selected)) {
                    selected_idx = i;
                    changed      = true;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        if (changed) {
            selected_value = index_items[selected_idx];
        }
        return changed;
    }

    template <typename T>
    inline bool combo_box(const char *name, const char *const items[], int item_count, T &selected)
    {
        bool changed = false;

        if (ImGui::BeginCombo(name, items[(int)selected])) {
            for (int i = 0; i < item_count; ++i) {
                bool is_selected = ((T)i == selected);

                if (ImGui::Selectable(items[i], is_selected)) {
                    selected = (T)i;
                    changed  = true;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }

            ImGui::EndCombo();
        }

        return changed;
    }

}

}