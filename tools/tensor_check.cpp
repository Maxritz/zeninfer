#include "zen/gguf.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
int main(int argc,char**argv){
    if(argc!=2){std::cerr<<"usage: zen_tensor_check model.gguf\n";return 2;}
    auto report=zen::gguf::inspect(argv[1]);
    if(!report.ok){std::cerr<<"inspect: "<<report.error<<"\n";return 1;}
    std::size_t checked=0; std::size_t total=0;
    for(const auto& t:report.tensors){
        std::vector<float> values; std::string error;
        if(!zen::gguf::read_tensor_f32(argv[1],report,t.name,values,&error)){
            std::cerr<<"FAIL "<<t.name<<": "<<error<<"\n";return 1;
        }
        for(float x:values)if(!std::isfinite(x)){std::cerr<<"FAIL non-finite value in "<<t.name<<"\n";return 1;}
        ++checked; total+=values.size();
        std::cout<<"OK "<<t.name<<" elements="<<values.size()<<" type="<<t.ggml_type<<"\n";
    }
    std::cout<<"PASS tensors="<<checked<<" elements="<<total<<"\n";
    return 0;
}
