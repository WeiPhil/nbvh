#pragma once
#include <utility>
#include <vector>

template <class K, class V>
struct unordered_vector {
    typedef std::pair<K, V> value_type;
    std::vector<value_type> elements;

    typename std::vector<value_type>::iterator find(K const &k)
    {
        auto it = this->elements.begin();
        for (auto ite = this->elements.end(); it != ite; ++it)
            if (it->first == k)
                return it;
        return it;
    }
    typename std::vector<value_type>::const_iterator find(K const &k) const
    {
        auto it = this->elements.begin();
        for (auto ite = this->elements.end(); it != ite; ++it)
            if (it->first == k)
                return it;
        return it;
    }

    V &operator[](K const &k)
    {
        auto it = this->find(k);
        if (it != this->elements.end())
            return it->second;
        this->elements.push_back({k, V{}});
        return this->elements.back().second;
    }

    typename std::vector<value_type>::iterator begin()
    {
        return this->elements.begin();
    }
    typename std::vector<value_type>::iterator end()
    {
        return this->elements.end();
    }
    typename std::vector<value_type>::const_iterator begin() const
    {
        return this->elements.begin();
    }
    typename std::vector<value_type>::const_iterator end() const
    {
        return this->elements.end();
    }

    bool empty() const
    {
        return elements.empty();
    }
    void clear()
    {
        elements.clear();
    }
};
