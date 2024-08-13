#include "worker.hpp"


Worker::Worker() {}

void Worker::Start() {
    worker = std::thread([this] {
        while (!stop_requested) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this] {
                    return !task_queue.empty() || stop_requested;
                });

                if (stop_requested && task_queue.empty()) {
                    return;
                }
                task = task_queue.front();
                task_queue.pop();
            }
            if (task) {
                task();
            }
        }
    });
    worker.detach();
}

void Worker::Stop() {
    stop_requested = true;
    cv.notify_all();
    // maybe wait to finish
    //std::this_thread::sleep_for(std::chrono::seconds(1));
}

void Worker::Request(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        task_queue.push(task);
    }
    cv.notify_one();

}