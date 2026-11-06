#include "zen/tokenizer.h"
#include "zen/diagnostics.h"
#include <algorithm>
#include <limits>
#include <array>
#include <charconv>
#include <queue>
#include <utility>
namespace zen::tokenizer { namespace {
std::size_t utf8_len(unsigned char c){if((c&0x80)==0)return 1;if((c&0xe0)==0xc0)return 2;if((c&0xf0)==0xe0)return 3;if((c&0xf8)==0xf0)return 4;return 1;}
bool allowed_piece(std::uint32_t type){return type==1||type==4;}
struct Symbol{std::size_t start{},len{},group{};int prev{-1},next{-1};};
struct Bigram{int left{},right{};float score{};std::size_t size{};};
struct LowerPriority{bool operator()(const Bigram&a,const Bigram&b)const noexcept{if(a.score!=b.score)return a.score<b.score;return a.left>b.left;}};
#include "unicode_categories.inc"
bool in_ranges(std::uint32_t cp,const std::pair<std::uint32_t,std::uint32_t>* first,const std::pair<std::uint32_t,std::uint32_t>* last) noexcept {
    auto it=std::upper_bound(first,last,cp,[](std::uint32_t value,const auto& range){return value<range.first;});
    if(it==first)return false;
    --it;
    return cp<=it->second;
}
std::uint32_t read_cp(const std::string& s,std::size_t p,std::size_t& n) noexcept {
    const auto c=static_cast<unsigned char>(s[p]);n=std::min(utf8_len(c),s.size()-p);std::uint32_t cp=0;
    if(n==1)return c;
    if(n==2)cp=c&0x1f;else if(n==3)cp=c&0x0f;else cp=c&7;
    for(std::size_t i=1;i<n;++i)cp=(cp<<6)|(static_cast<unsigned char>(s[p+i])&0x3f);
    return cp;
}
enum class GptClass { letter, number, whitespace, other };
GptClass gpt_class(std::uint32_t cp) noexcept {
    if(in_ranges(cp,std::begin(kUnicodeLetterRanges),std::end(kUnicodeLetterRanges)))return GptClass::letter;
    if(in_ranges(cp,std::begin(kUnicodeNumberRanges),std::end(kUnicodeNumberRanges)))return GptClass::number;
    if(in_ranges(cp,std::begin(kUnicodeWhitespaceRanges),std::end(kUnicodeWhitespaceRanges)))return GptClass::whitespace;
    return GptClass::other;
}
std::vector<std::pair<std::size_t,std::size_t>> gpt2_pretoken_spans(const std::string& s) {
    struct Cp {std::size_t begin,end;std::uint32_t value;GptClass cls;};
    std::vector<Cp> cps;
    for(std::size_t p=0;p<s.size();){std::size_t n=1;auto cp=read_cp(s,p,n);cps.push_back({p,p+n,cp,gpt_class(cp)});p+=n;}
    std::vector<std::pair<std::size_t,std::size_t>> spans;
    auto is_ascii=[&](std::size_t i,char c){return i<cps.size()&&cps[i].value==static_cast<unsigned char>(c);};
    auto lower_ascii=[](std::uint32_t c){return c>='A'&&c<='Z'?c+('a'-'A'):c;};
    for(std::size_t i=0;i<cps.size();) {
        const std::size_t start=i;std::size_t end=i;
        // GPT-2 contraction alternatives are case-insensitive and take precedence.
        if(cps[i].value=='\'') {
            static constexpr const char* suffixes[]={"re","ve","ll","s","t","m","d"};
            for(const char* suffix:suffixes){std::size_t j=i+1,k=0;while(suffix[k]&&j<cps.size()&&lower_ascii(cps[j].value)==static_cast<unsigned char>(suffix[k])){++j;++k;}if(!suffix[k]){end=j;break;}}
        }
        if(end==i) {
            std::size_t content=i;
            if(is_ascii(i,' ')&&i+1<cps.size()&&cps[i+1].cls!=GptClass::whitespace)content=i+1;
            if(content<cps.size()&&cps[content].cls==GptClass::letter){end=content+1;while(end<cps.size()&&cps[end].cls==GptClass::letter)++end;}
            else if(content<cps.size()&&cps[content].cls==GptClass::number){end=content+1;while(end<cps.size()&&cps[end].cls==GptClass::number)++end;}
            else if(content<cps.size()&&cps[content].cls==GptClass::other){end=content+1;while(end<cps.size()&&cps[end].cls==GptClass::other)++end;while(end<cps.size()&&(cps[end].value=='\r'||cps[end].value=='\n'))++end;}
            else {
                std::size_t run_end=i+1,last_newline=std::numeric_limits<std::size_t>::max();
                while(run_end<cps.size()&&cps[run_end].cls==GptClass::whitespace){if(cps[run_end].value=='\r'||cps[run_end].value=='\n')last_newline=run_end;++run_end;}
                // GPT-2's \s*[\r\n]+ alternative consumes through the final
                // newline in a whitespace run; trailing horizontal whitespace
                // is left for the following whitespace alternative.
                end=last_newline==std::numeric_limits<std::size_t>::max()?run_end:last_newline+1;
            }
        }
        if(end<=start)end=start+1;
        spans.emplace_back(cps[start].begin,cps[end-1].end);i=end;
    }
    return spans;
}
}
bool ModelTokenizer::load(const model::ModelConfig& c,std::string* error){
    zen::diag::Scope trace("tokenizer.load",zen::diag::Phase::io);
    auto fail=[&](const char* e){if(error)*error=e;tokens_.clear();token_to_id_.clear();return false;};
    auto tokenizer_model=c.metadata.find("tokenizer.ggml.model");
    if(tokenizer_model==c.metadata.end()||(tokenizer_model->second!="llama"&&tokenizer_model->second!="gpt2"))return fail("Supported GGUF tokenizer models are llama and gpt2");
    gpt2_bpe_=tokenizer_model->second=="gpt2";gpt2_decoder_.clear();
    if(c.tokenizer_tokens.empty())return fail("GGUF has no tokenizer.ggml.tokens array");
    if(!c.tokenizer_scores.empty()&&c.tokenizer_scores.size()!=c.tokenizer_tokens.size())return fail("Tokenizer score count does not match vocabulary");
    if(!c.tokenizer_token_types.empty()&&c.tokenizer_token_types.size()!=c.tokenizer_tokens.size())return fail("Tokenizer token_type count does not match vocabulary");
    tokens_=c.tokenizer_tokens;scores_=c.tokenizer_scores;types_=c.tokenizer_token_types;
    if(scores_.empty())scores_.assign(tokens_.size(),0.0f);
    if(types_.empty())types_.assign(tokens_.size(),1);
    if(gpt2_bpe_)for(std::size_t i=0;i<scores_.size();++i)scores_[i]=-static_cast<float>(i);
    bos_=c.bos_token_id;eos_=c.eos_token_id;unk_=c.unknown_token_id;add_space_prefix_=c.add_space_prefix;
    token_to_id_.clear();byte_to_id_.assign(256,-1);
    for(std::uint32_t i=0;i<tokens_.size();++i){
        if(allowed_piece(types_[i]))token_to_id_.emplace(tokens_[i],i);
        if(types_[i]==6&&tokens_[i].size()==6&&tokens_[i].compare(0,3,"<0x")==0&&tokens_[i][5]=='>'){
            unsigned value{};auto res=std::from_chars(tokens_[i].data()+3,tokens_[i].data()+5,value,16);
            if(res.ec==std::errc{}&&res.ptr==tokens_[i].data()+5&&value<256)byte_to_id_[value]=static_cast<int>(i);
        }
    }
    if(gpt2_bpe_){std::vector<std::uint32_t> bytes,codepoints;for(unsigned b=33;b<=126;++b)bytes.push_back(b);for(unsigned b=161;b<=172;++b)bytes.push_back(b);for(unsigned b=174;b<=255;++b)bytes.push_back(b);codepoints=bytes;std::uint32_t extra=0;for(unsigned b=0;b<256;++b)if(std::find(bytes.begin(),bytes.end(),b)==bytes.end()){bytes.push_back(b);codepoints.push_back(256+extra++);}for(std::size_t i=0;i<bytes.size();++i)gpt2_decoder_[codepoints[i]]=static_cast<std::uint8_t>(bytes[i]);}
    if(bos_>=tokens_.size()||eos_>=tokens_.size()||unk_>=tokens_.size())return fail("Tokenizer special-token ID is outside vocabulary");
    if(error)error->clear();
    return true;
}
bool ModelTokenizer::encode(const std::string& input,std::vector<std::uint32_t>& ids,bool add_bos,bool add_eos)const{
    zen::diag::Scope trace("tokenizer.encode",zen::diag::Phase::compute);
    if(tokens_.empty())return false;
    // SentencePiece's Llama tokenizer is a unigram language model. It must use
    // Viterbi segmentation over token scores, not BPE-style pair merging.
    if(!gpt2_bpe_){
        const std::string marker="\xE2\x96\x81";
        std::string normalized;normalized.reserve(input.size()+marker.size());
        bool pending_space=false;
        for(unsigned char ch:input){
            if(ch==' '){
                if(!normalized.empty())pending_space=true;
                continue;
            }
            if(pending_space){normalized+=marker;pending_space=false;}
            normalized.push_back(static_cast<char>(ch));
        }
        if(add_space_prefix_&&!normalized.empty())normalized.insert(0,marker);
        std::vector<std::uint32_t> result;
        if(add_bos)result.push_back(bos_);
        if(normalized.empty()){if(add_eos) result.push_back(eos_); ids.swap(result); return true;}
        struct Path { double score=-std::numeric_limits<double>::infinity(); std::size_t previous=0; std::vector<std::uint32_t> emitted; bool valid=false; };
        std::vector<Path> best(normalized.size()+1);best[0].score=0.0;best[0].valid=true;
        std::size_t max_piece_bytes=1;for(const auto& piece:tokens_)max_piece_bytes=std::max(max_piece_bytes,piece.size());
        std::vector<std::size_t> boundaries;boundaries.reserve(normalized.size()+1);boundaries.push_back(0);
        for(std::size_t pos=0;pos<normalized.size();){const auto n=std::min(utf8_len(static_cast<unsigned char>(normalized[pos])),normalized.size()-pos);pos+=n;boundaries.push_back(pos);}
        for(std::size_t bi=0;bi+1<boundaries.size();++bi){
            const std::size_t pos=boundaries[bi];if(!best[pos].valid)continue;
            for(std::size_t bj=bi+1;bj<boundaries.size()&&boundaries[bj]-pos<=max_piece_bytes;++bj){
                const std::size_t end=boundaries[bj];auto it=token_to_id_.find(normalized.substr(pos,end-pos));if(it==token_to_id_.end())continue;
                const auto id=it->second;const double candidate=best[pos].score+static_cast<double>(scores_[id]);
                if(!best[end].valid||candidate>best[end].score){best[end].score=candidate;best[end].previous=pos;best[end].emitted.assign(1,id);best[end].valid=true;}
            }
            // Byte fallback is a single transition per UTF-8 codepoint. This
            // mirrors SentencePiece's byte_fallback rather than emitting <unk>.
            const std::size_t cp_end=boundaries[bi+1];std::vector<std::uint32_t> fallback;double fallback_score=best[pos].score;
            bool has_bytes=true;for(std::size_t k=pos;k<cp_end;++k){const int id=byte_to_id_[static_cast<unsigned char>(normalized[k])];if(id<0){has_bytes=false;break;}fallback.push_back(static_cast<std::uint32_t>(id));fallback_score+=scores_[static_cast<std::size_t>(id)];}
            if(!has_bytes){fallback.assign(1,unk_);fallback_score+=scores_[unk_];}
            const std::size_t fallback_end=cp_end;
            if(!best[fallback_end].valid||fallback_score>best[fallback_end].score){best[fallback_end].score=fallback_score;best[fallback_end].previous=pos;best[fallback_end].emitted=std::move(fallback);best[fallback_end].valid=true;}
        }
        if(!best.back().valid)return false;
        std::vector<std::vector<std::uint32_t>> chunks;for(std::size_t pos=normalized.size();pos>0;){const auto& path=best[pos];if(!path.valid||path.previous>=pos)return false;chunks.push_back(path.emitted);pos=path.previous;}
        for(auto it=chunks.rbegin();it!=chunks.rend();++it)result.insert(result.end(),it->begin(),it->end());
        if(add_eos) result.push_back(eos_);
        ids.swap(result);
        return true;
    }
    std::string text;text.reserve(input.size()+8);std::vector<std::size_t> text_groups;const std::string space="\xE2\x96\x81";
    auto append_text=[&](const std::string& piece,std::size_t group){text+=piece;text_groups.insert(text_groups.end(),piece.size(),group);};
    if(gpt2_bpe_){
        std::vector<std::uint32_t> bytes,codepoints;for(unsigned b=33;b<=126;++b)bytes.push_back(b);for(unsigned b=161;b<=172;++b)bytes.push_back(b);for(unsigned b=174;b<=255;++b)bytes.push_back(b);codepoints=bytes;
        std::uint32_t extra=0;for(unsigned b=0;b<256;++b)if(std::find(bytes.begin(),bytes.end(),b)==bytes.end()){bytes.push_back(b);codepoints.push_back(256+extra++);}
        std::array<std::uint32_t,256> encoder{};for(std::size_t i=0;i<bytes.size();++i)encoder[bytes[i]]=codepoints[i];
        auto encode_cp=[](std::uint32_t cp){std::string out;if(cp<=0x7f)out+=char(cp);else if(cp<=0x7ff){out+=char(0xc0|(cp>>6));out+=char(0x80|(cp&63));}else if(cp<=0xffff){out+=char(0xe0|(cp>>12));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}else{out+=char(0xf0|(cp>>18));out+=char(0x80|((cp>>12)&63));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}return out;};
        // GPT-2 byte-level BPE uses regex-defined pre-token spans. Unicode
        // L/N/Whitespace categories are generated static tables, avoiding ICU
        // as a platform-specific runtime dependency.
        std::vector<std::size_t> byte_group(input.size());std::size_t group=0;
        for(const auto& span:gpt2_pretoken_spans(input)){for(std::size_t j=span.first;j<span.second;++j)byte_group[j]=group;++group;}
        for(std::size_t i=0;i<input.size();++i)append_text(encode_cp(encoder[static_cast<unsigned char>(input[i])]),byte_group[i]);
    }else{
        if(add_space_prefix_&&!input.empty())append_text(space,0);
        for(char c:input){if(c==' '||c=='\t'||c=='\n'||c=='\r')append_text(space,0);else append_text(std::string(1,c),0);}
    }
    std::vector<Symbol> symbols;
    for(std::size_t off=0;off<text.size();){const auto n=std::min(utf8_len(static_cast<unsigned char>(text[off])),text.size()-off);
        Symbol sym;sym.start=off;sym.len=n;sym.group=text_groups[off];sym.prev=symbols.empty()?-1:static_cast<int>(symbols.size()-1);sym.next=-1;
        if(!symbols.empty())symbols.back().next=static_cast<int>(symbols.size());
        symbols.push_back(sym);off+=n;
    }
    std::priority_queue<Bigram,std::vector<Bigram>,LowerPriority> queue;
    std::unordered_map<std::string,std::pair<int,int>> reverse_merge;
    auto add_pair=[&](int left,int right){
        if(left<0||right<0||symbols[left].len==0||symbols[right].len==0)return;
        const auto& a=symbols[left];const auto& b=symbols[right];if(a.next!=right||a.group!=b.group)return;
        const std::string combined=text.substr(a.start,a.len+b.len);auto it=token_to_id_.find(combined);if(it==token_to_id_.end())return;
        const auto id=it->second;queue.push({left,right,scores_[id],combined.size()});reverse_merge[combined]={left,right};
    };
    for(std::size_t i=1;i<symbols.size();++i)add_pair(static_cast<int>(i-1),static_cast<int>(i));
    while(!queue.empty()){
        const auto m=queue.top();queue.pop();auto& left=symbols[m.left];auto& right=symbols[m.right];
        if(left.len==0||right.len==0||left.next!=m.right||left.len+right.len!=m.size)continue;
        left.len+=right.len;right.len=0;left.next=right.next;if(right.next>=0)symbols[right.next].prev=m.left;
        add_pair(left.prev,m.left);add_pair(m.left,left.next);
    }
    std::vector<std::uint32_t> result;if(add_bos)result.push_back(bos_);
    if(!symbols.empty()){
        for(int root=0;root!=-1;root=symbols[root].next){std::vector<int> todo{root};
            while(!todo.empty()){
                const int idx=todo.back();todo.pop_back();const auto& sym=symbols[idx];const std::string piece=text.substr(sym.start,sym.len);
                auto found=token_to_id_.find(piece);if(found!=token_to_id_.end()){result.push_back(found->second);continue;}
                auto split=reverse_merge.find(piece);if(split!=reverse_merge.end()){todo.push_back(split->second.second);todo.push_back(split->second.first);continue;}
                if(gpt2_bpe_){for(std::size_t p=0;p<piece.size();){const auto n=std::min(utf8_len(static_cast<unsigned char>(piece[p])),piece.size()-p);std::uint32_t cp=0;for(std::size_t j=0;j<n;++j){const auto c=static_cast<unsigned char>(piece[p+j]);if(j==0){if(n==1)cp=c;else if(n==2)cp=c&0x1f;else if(n==3)cp=c&0x0f;else cp=c&7;}else cp=(cp<<6)|(c&0x3f);}auto decoded=gpt2_decoder_.find(cp);if(decoded!=gpt2_decoder_.end()){const int id=byte_to_id_[decoded->second];result.push_back(id>=0?static_cast<std::uint32_t>(id):unk_);}else result.push_back(unk_);p+=n;}}
                else for(unsigned char byte:piece){const int id=byte_to_id_[byte];result.push_back(id>=0?static_cast<std::uint32_t>(id):unk_);}
            }
        }
    }
    if(add_eos)result.push_back(eos_);
    ids.swap(result);return true;
}
bool ModelTokenizer::decode(const std::vector<std::uint32_t>& ids,std::string& out,bool skip_special)const{
    zen::diag::Scope trace("tokenizer.decode",zen::diag::Phase::compute);
    if(tokens_.empty())return false;
    std::string result;const std::string escaped="\xE2\x96\x81";
    for(auto id:ids){if(id>=tokens_.size())return false;const auto type=types_[id];
        if(type==6){if(tokens_[id].size()==6){unsigned byte{};auto r=std::from_chars(tokens_[id].data()+3,tokens_[id].data()+5,byte,16);if(r.ec==std::errc{}&&byte<256)result.push_back(static_cast<char>(byte));}continue;}
        if(gpt2_bpe_){if((type==3||type==2)&&skip_special)continue;const auto& piece=tokens_[id];for(std::size_t p=0;p<piece.size();){const auto n=std::min(utf8_len(static_cast<unsigned char>(piece[p])),piece.size()-p);std::uint32_t cp=0;for(std::size_t j=0;j<n;++j){const auto c=static_cast<unsigned char>(piece[p+j]);if(j==0){if(n==1)cp=c;else if(n==2)cp=c&0x1f;else if(n==3)cp=c&0x0f;else cp=c&7;}else cp=(cp<<6)|(c&0x3f);}auto it=gpt2_decoder_.find(cp);if(it!=gpt2_decoder_.end())result.push_back(static_cast<char>(it->second));else result.append(piece,p,n);p+=n;}continue;}
        if((type==3||type==2)&&skip_special)continue;
        std::string piece=tokens_[id];std::size_t pos=0;
        while((pos=piece.find(escaped,pos))!=std::string::npos){piece.replace(pos,escaped.size()," ");++pos;}
        result+=piece;
    }
    if(!result.empty()&&result.front()==' ')result.erase(result.begin());
    out.swap(result);return true;
}
bool ModelTokenizer::same_vocabulary(const ModelTokenizer& other)const noexcept{return gpt2_bpe_==other.gpt2_bpe_&&tokens_==other.tokens_&&types_==other.types_&&bos_==other.bos_&&eos_==other.eos_&&unk_==other.unk_;}

}
