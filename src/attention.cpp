#include "zen/attention.h"
#include <cmath>
namespace zen::attention {
float alibi_head_slope(std::uint32_t head,std::uint32_t heads) noexcept {
    if(!heads||head>=heads)return 0.0f;
    std::uint32_t power=1;
    while(power<=heads/2)power<<=1;
    const float base=std::pow(2.0f,-8.0f/static_cast<float>(power));
    if(head<power)return std::pow(base,static_cast<float>(head+1));
    const float extra=std::pow(2.0f,-4.0f/static_cast<float>(power));
    return std::pow(extra,static_cast<float>(2*(head-power)+1));
}
float alibi_bias(std::uint32_t head,std::uint32_t heads,std::uint32_t q,std::uint32_t k) noexcept {
    if(k>q)return -INFINITY;
    return -alibi_head_slope(head,heads)*static_cast<float>(q-k);
}
}
