#include "zen/compute_pool.h"
#include <algorithm>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace zen {
struct ComputePool::Impl {
    std::mutex mutex;
    std::condition_variable ready;
    bool stopping{};
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> workers;
    explicit Impl(unsigned n) {
        workers.reserve(n);
        for (unsigned i=0;i<n;++i) workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready.wait(lock,[this]{return stopping||!tasks.empty();});
                    if(stopping&&tasks.empty())return;
                    task=std::move(tasks.front());tasks.pop();
                }
                task();
            }
        });
    }
    ~Impl() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        ready.notify_all();
        for(auto& worker:workers)if(worker.joinable())worker.join();
    }
};
ComputePool::ComputePool() {
    const unsigned hw=std::max(1u,std::thread::hardware_concurrency());
    const unsigned total=std::min(32u,hw);
    impl_=new Impl(total>1?total-1:0);
}
ComputePool::~ComputePool(){delete impl_;}
ComputePool& ComputePool::instance(){static ComputePool pool;return pool;}
unsigned ComputePool::worker_count()const noexcept{return static_cast<unsigned>(impl_->workers.size()+1);}
void ComputePool::parallel_for(std::size_t count,unsigned requested,
    const std::function<void(std::size_t,std::size_t)>& fn) {
    if(!count)return;
    const unsigned workers=std::max(1u,std::min<unsigned>(requested?requested:worker_count(),
        static_cast<unsigned>(std::min<std::size_t>(count,worker_count()))));
    if(workers==1){fn(0,count);return;}
    struct Job {std::mutex mutex;std::condition_variable done;std::size_t pending{};std::exception_ptr error;};
    auto job=std::make_shared<Job>();job->pending=workers-1;
    auto run=[job,fn](std::size_t first,std::size_t last){try{fn(first,last);}catch(...){std::lock_guard<std::mutex> l(job->mutex);if(!job->error)job->error=std::current_exception();}};
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for(unsigned i=1;i<workers;++i){
            const auto first=count*i/workers,last=count*(i+1)/workers;
            impl_->tasks.emplace([job,run,first,last]{run(first,last);{std::lock_guard<std::mutex> l(job->mutex);--job->pending;}job->done.notify_one();});
        }
    }
    impl_->ready.notify_all();
    run(0,count/workers);
    std::unique_lock<std::mutex> lock(job->mutex);job->done.wait(lock,[&]{return job->pending==0;});
    if(job->error)std::rethrow_exception(job->error);
}
}
