#include "zen/chat_template.h"
namespace zen::chat {
bool format_legacy_chat(const std::string& tmpl,const std::string& architecture,
                        const std::vector<http::ChatMessage>& messages,std::string& prompt,
                        std::string* error) {
    const bool llama3=tmpl.find("<|start_header_id|>")!=std::string::npos;
    const bool llama2=tmpl.find("[INST]")!=std::string::npos;
    const bool chatml=tmpl.find("<|im_start|>")!=std::string::npos||architecture=="qwen2";
    if(llama2){
        std::string system;for(const auto& m:messages)if(m.role=="system")system+=m.content+"\n";
        bool first=true;
        for(const auto& m:messages){
            if(m.role=="system")continue;
            if(m.role=="user"){
                prompt+="[INST] ";
                if(first&&!system.empty())prompt+="<<SYS>>\n"+system+"<</SYS>>\n\n";
                prompt+=m.content+" [/INST]";first=false;
            } else if(m.role=="assistant")prompt+=m.content+" ";
            else {if(error) *error="Unsupported chat role: "+m.role;
                return false;}
        }
    } else {
        for(const auto& m:messages){
            if(m.role!="system"&&m.role!="user"&&m.role!="assistant"){
                if(error) *error="Unsupported chat role: "+m.role;
                return false;
            }
            if(llama3)prompt+="<|start_header_id|>"+m.role+"<|end_header_id|>\n\n"+m.content+"<|eot_id|>\n";
            else if(chatml)prompt+="<|im_start|>"+m.role+"\n"+m.content+"<|im_end|>\n";
            else prompt+="### "+m.role+":\n"+m.content+"\n";
        }
    }
    if(!messages.empty()){
        if(llama3)prompt+="<|start_header_id|>assistant<|end_header_id|>\n\n";
        else if(chatml)prompt+="<|im_start|>assistant\n";
        else if(!llama2)prompt+="### assistant:\n";
    }
    if(prompt.empty()){if(error)*error="Chat messages did not produce a prompt";return false;}
    return true;
}
}
