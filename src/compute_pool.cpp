#include "zen/compute_pool.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace zen {
namespace detail {
inline void cpu_pause() noexcept {
#if defined(_MSC_VER)
    _mm_pause();  // YieldProcessor lives in winnt.h, not intrin.h
#elif defined(__i386__)||defined(__x86_64__)||defined(_M_IX86)||defined(_M_X64)
    __builtin_ia32_pause();
#endif
}
}
namespace {
constexpr std::chrono::microseconds kSpinWindow{1500};
}
struct ComputePool::Impl {
    std::mutex mutex;
    std::condition_variable ready;
    bool stopping{};
    std::size_t sleeping{}; // workers blocked in ready.wait, guarded by mutex
    std::queue<std::function<void()>> tasks;
    std::vector<std::thread> workers;
    explicit Impl(unsigned n) {
        workers.reserve(n);
        for (unsigned i=0;i<n;++i) workers.emplace_back([this] {
            // ZI-P2 launch elimination. Decode dispatches ~113 pool jobs per
            // token with sub-millisecond gaps; waking a blocked Windows thread
            // measured ~108us per dispatch (~12ms/token of pure launch cost).
            // A worker that just finished a job spins briefly before sleeping,
            // so back-to-back dispatches are picked up in microseconds.
            // Workers that never received a job do NOT spin: the planner uses
            // fewer workers than the pool size, and idle spinners starve other
            // threads (prefetch, caller) of cores on a saturated machine.
            bool worked=false;
            for (;;) {
                std::function<void()> task;
                bool have=false, quit=false;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    auto try_pop=[&]{
                        if(stopping&&tasks.empty()){quit=true;return;}
                        if(!tasks.empty()){task=std::move(tasks.front());tasks.pop();have=true;}
                    };
                    try_pop();
                    if(!have&&!quit&&worked){
                        // Spin only after real work; mutex released while pausing.
                        const auto until=std::chrono::steady_clock::now()+kSpinWindow;
                        while(!have&&!quit&&std::chrono::steady_clock::now()<until){
                            lock.unlock();
                            for(int k=0;k<256;++k)detail::cpu_pause();
                            lock.lock();
                            try_pop();
                        }
                    }
                    if(!have&&!quit){
                        ++sleeping;
                        ready.wait(lock,[this]{return stopping||!tasks.empty();});
                        --sleeping;
                        try_pop();
                    }
                }
                if(quit)return;
                if(have){task();worked=true;}
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
    unsigned wakes=0;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for(unsigned i=1;i<workers;++i){
            const auto first=count*i/workers,last=count*(i+1)/workers;
            impl_->tasks.emplace([job,run,first,last]{run(first,last);{std::lock_guard<std::mutex> l(job->mutex);--job->pending;}job->done.notify_one();});
        }
        // ZI-P2: notify only workers that are actually blocked. Spinning workers
        // claim tasks without any syscall; waking extra sleepers that then find
        // an empty queue costs a kernel round trip each (notify_all used to rouse
        // all 31 workers for every 15-task job).
        wakes=impl_->sleeping;
    }
    const unsigned enqueued=workers-1;
    for(unsigned n=0;n<std::min(wakes,enqueued);++n)impl_->ready.notify_one();
    run(0,count/workers);
    std::unique_lock<std::mutex> lock(job->mutex);job->done.wait(lock,[&]{return job->pending==0;});
    if(job->error)std::rethrow_exception(job->error);
}
}
