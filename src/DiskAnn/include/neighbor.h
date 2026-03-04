// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;

    Neighbor() = default;

    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
};

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
    {
    }

    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
    {
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }

    Neighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }

    bool has_unexpanded_node() const
    {
        return _cur < _size;
    }

    size_t size() const
    {
        return _size;
    }

    size_t capacity() const
    {
        return _capacity;
    }

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    // Set new capacity for resumable search. When expanding capacity,
    // the queue can accept more candidates. When shrinking, excess items are dropped.
    // This method preserves the expansion state of existing items.
    void set_capacity(size_t new_capacity)
    {
        if (new_capacity + 1 > _data.size())
        {
            _data.resize(new_capacity + 1);
        }
        _capacity = new_capacity;
        // If size exceeds new capacity, truncate
        if (_size > _capacity)
        {
            _size = _capacity;
        }
        // Ensure _cur doesn't exceed _size
        if (_cur > _size)
        {
            _cur = _size;
        }
    }

    // Get current cursor position (for state serialization)
    size_t get_cur() const
    {
        return _cur;
    }

    // Set cursor position (for state restoration)
    void set_cur(size_t cur)
    {
        _cur = cur;
        // Ensure _cur doesn't exceed _size
        if (_cur > _size)
        {
            _cur = _size;
        }
    }

    // Reset cursor to first unexpanded node
    // This scans through the data to find the first unexpanded position
    void reset_cursor()
    {
        _cur = 0;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
    }

    // Get all data for state serialization
    const std::vector<Neighbor>& get_data() const
    {
        return _data;
    }

    // Restore state from serialized data
    void restore_state(const std::vector<Neighbor>& data, size_t size, size_t capacity, size_t cur)
    {
        _data = data;
        _size = size;
        _capacity = capacity;
        _cur = cur;
        if (_data.size() < _capacity + 1)
        {
            _data.resize(_capacity + 1);
        }
    }

    Neighbor &operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
    }

  private:
    size_t _size, _capacity, _cur;
    std::vector<Neighbor> _data;
};

} // namespace diskann
