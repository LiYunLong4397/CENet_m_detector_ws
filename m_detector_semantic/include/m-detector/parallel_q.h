#ifndef PARALLEL_Q_H
#define PARALLEL_Q_H

#include <deque>
#include <mutex>
#include <condition_variable>

template<typename T>
class PARALLEL_Q
{
private:
    std::deque<T> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;
    size_t max_size;
    
public:
    PARALLEL_Q(size_t max_size = 1000000) : max_size(max_size) {}
    
    void push(const T& item)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.size() >= max_size) {
            queue.pop_front();
        }
        queue.push_back(item);
        cv.notify_one();
    }
    
    bool pop(T& item)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (queue.empty()) {
            return false;
        }
        item = queue.front();
        queue.pop_front();
        return true;
    }
    
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }
    
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }
    
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        queue.clear();
    }
};

#endif // PARALLEL_Q_H