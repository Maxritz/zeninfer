#include "zen/resource_governor.h"
#include <algorithm>
#include <set>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <sched.h>
#include <fstream>
#include <string>
#endif
namespace zen {
ResourceSnapshot ResourceGovernor::snapshot() const noexcept {
 ResourceSnapshot s{};
#if defined(_WIN32)
 SYSTEM_INFO si{}; GetSystemInfo(&si); s.logical_cpus=si.dwNumberOfProcessors;
 DWORD_PTR process_mask=0,system_mask=0;
 if(GetProcessAffinityMask(GetCurrentProcess(),&process_mask,&system_mask)){unsigned allowed=0;for(DWORD_PTR bits=process_mask;bits;bits&=(bits-1))++allowed;if(allowed)s.logical_cpus=allowed;}
 DWORD bytes=0;GetLogicalProcessorInformationEx(RelationProcessorCore,nullptr,&bytes);
 if(bytes){std::vector<unsigned char> buf(bytes);if(GetLogicalProcessorInformationEx(RelationProcessorCore,reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),&bytes)){DWORD off=0;while(off<bytes){auto*e=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()+off);if(e->Size==0)break;++s.physical_cpus;off+=e->Size;}}}
 if(!s.physical_cpus)s.physical_cpus=std::max(1u,s.logical_cpus/2);
 MEMORYSTATUSEX m{};m.dwLength=sizeof(m);if(GlobalMemoryStatusEx(&m)){s.total_ram_bytes=m.ullTotalPhys;s.available_ram_bytes=m.ullAvailPhys;}
 s.process_memory_limit_bytes=s.total_ram_bytes;
#else
 long n=sysconf(_SC_NPROCESSORS_ONLN);s.logical_cpus=n>0?unsigned(n):1u;
 cpu_set_t set;CPU_ZERO(&set);if(sched_getaffinity(0,sizeof(set),&set)==0){int a=CPU_COUNT(&set);if(a>0)s.logical_cpus=unsigned(a);}
 std::set<std::pair<int,int>> cores;
 for(int cpu=0;cpu<CPU_SETSIZE;++cpu){if(!CPU_ISSET(cpu,&set))continue;std::ifstream cf("/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/core_id");std::ifstream pf("/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/physical_package_id");int core=-1,pkg=0;if(cf>>core){pf>>pkg;cores.emplace(pkg,core);}}
 s.physical_cpus=cores.empty()?std::max(1u,s.logical_cpus/2):static_cast<unsigned>(cores.size());
 long p=sysconf(_SC_PHYS_PAGES),z=sysconf(_SC_PAGE_SIZE),a=sysconf(_SC_AVPHYS_PAGES);
 if(p>0&&z>0)s.total_ram_bytes=std::uint64_t(p)*std::uint64_t(z);
 if(a>0&&z>0)s.available_ram_bytes=std::uint64_t(a)*std::uint64_t(z);
 s.process_memory_limit_bytes=s.total_ram_bytes;
 std::ifstream f("/sys/fs/cgroup/memory.max");std::string v;if(f>>v&&v!="max")try{s.process_memory_limit_bytes=std::min<std::uint64_t>(s.process_memory_limit_bytes,static_cast<std::uint64_t>(std::stoull(v)));}catch(...){}
 std::ifstream cf("/sys/fs/cgroup/memory.current");std::uint64_t used=0;
 if(cf>>used){auto cgroup_free=s.process_memory_limit_bytes>used?s.process_memory_limit_bytes-used:0;s.available_ram_bytes=std::min(s.available_ram_bytes,cgroup_free);}
 std::ifstream cpu_file("/sys/fs/cgroup/cpu.max");std::uint64_t quota=0,period=0;std::string quota_text;
 if(cpu_file>>quota_text>>period&&quota_text!="max"){try{quota=std::stoull(quota_text);if(period){unsigned quota_cpus=static_cast<unsigned>(std::max<std::uint64_t>(1,(quota+period-1)/period));s.logical_cpus=std::min(s.logical_cpus,quota_cpus);}}catch(...){}}
#endif
 if(!s.logical_cpus) s.logical_cpus=1;
 if(!s.physical_cpus) s.physical_cpus=std::max(1u,s.logical_cpus/2);
 s.physical_cpus=std::min(s.physical_cpus,s.logical_cpus);
 if(!s.available_ram_bytes) {
#if defined(_WIN32)
  s.available_ram_bytes=s.total_ram_bytes/4;
#else
  std::ifstream current("/sys/fs/cgroup/memory.current");std::uint64_t used_now=0;
  std::ifstream limit_file("/sys/fs/cgroup/memory.max");std::uint64_t limit_now=0;std::string limit_text;
  if(current>>used_now&&limit_file>>limit_text&&limit_text!="max"){try{limit_now=std::stoull(limit_text);}catch(...){}}
  if(!(limit_now&&used_now>=limit_now))s.available_ram_bytes=s.total_ram_bytes/4;
#endif
 }
 if(!s.process_memory_limit_bytes) s.process_memory_limit_bytes=s.total_ram_bytes;
 return s;
}
ResourcePolicy ResourceGovernor::policy(const ResourceSnapshot&s)const noexcept{
 ResourcePolicy p{};
 const unsigned c=std::max(1u,std::min(s.logical_cpus,s.physical_cpus));
 const auto usable=std::min(s.available_ram_bytes,s.process_memory_limit_bytes);
 // Concurrency is constrained by both CPU capacity and current free-memory headroom.
 p.worker_threads=usable<(2ull<<30)?1u:usable<(4ull<<30)?std::max(1u,c/2u):(c<=2?1u:c);
 const auto reserve=std::max<std::uint64_t>(512ull<<20,usable/5);
 const auto managed=usable>reserve?usable-reserve:usable/2;
 p.cache_budget_bytes=managed*60/100;
 p.transient_budget_bytes=managed-p.cache_budget_bytes;
 if(usable>=32ull<<30){p.memory_tier="abundant";p.prefetch_workers=std::min(4u,std::max(1u,c/8));}
 else if(usable>=12ull<<30){p.memory_tier="balanced";p.prefetch_workers=std::min(2u,std::max(1u,c/8));}
 else{p.memory_tier="constrained";p.prefetch_workers=usable>=(4ull<<30)&&c>=4?1:0;}
 return p;
}
ResourcePolicy ResourceGovernor::current_policy()const noexcept{return policy(snapshot());}
}
