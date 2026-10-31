#include "zen/quant.h"
#include "zen/compute_pool.h"
#include "zen/attention.h"
#include "zen/execution_planner.h"
#include "zen/model_ir.h"
#include "zen/gguf.h"
#include "zen/tokenizer.h"
#include "zen/model_runtime.h"
#include "zen/chat_template.h"
#include "zen/openai_server.h"
#include "data/fp16_reference_corpus.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <stdexcept>
#include <vector>
static int failures=0;
#define CHECK(x) do{if(!(x)){std::cerr<<"FAIL line "<<__LINE__<<": " #x "\n";++failures;}}while(0)
int main(){
    using namespace zen;
    CHECK(model::identify_architecture("llama")==model::Architecture::llama);
    CHECK(model::identify_architecture("qwen3moe")==model::Architecture::qwen);
    CHECK(model::identify_architecture("bitnet-b1.58")==model::Architecture::bitnet);
    CHECK(model::identify_architecture("bonsai")==model::Architecture::bonsai);
    {
        model::ModelConfig c;c.architecture=model::Architecture::llama;c.architecture_name="llama";
        std::string adapter_error;CHECK(runtime::create_model_runtime(c,&adapter_error)!=nullptr);
        c.architecture=model::Architecture::qwen;c.architecture_name="qwen2";CHECK(runtime::create_model_runtime(c,&adapter_error)!=nullptr);
        c.architecture_name="qwen3";CHECK(runtime::create_model_runtime(c,&adapter_error)!=nullptr);
        c.architecture_name="qwen3moe";CHECK(runtime::create_model_runtime(c,&adapter_error)==nullptr);
        CHECK(adapter_error.find("qwen3moe")!=std::string::npos);
        c.architecture=model::Architecture::bitnet;c.architecture_name="bitnet";CHECK(runtime::create_model_runtime(c,&adapter_error)!=nullptr); // ZI-003 GREEN: BitNet adapter registered
        c.architecture=model::Architecture::bonsai;c.architecture_name="bonsai";CHECK(runtime::create_model_runtime(c,&adapter_error)==nullptr);
    }
    CHECK(model::infer_tensor_role("blk.0.attn_q.weight")==model::TensorRole::query);
    CHECK(model::infer_tensor_role("model.layers.0.mlp.gate_proj.weight")==model::TensorRole::ffn_gate);
    CHECK(model::infer_tensor_role("blk.0.ffn_gate_inp.weight")==model::TensorRole::router);
    CHECK(model::infer_tensor_role("blk.0.ffn_gate_exps.weight")==model::TensorRole::expert_gate);
    CHECK(model::infer_tensor_role("blk.0.ffn_up_exps.weight")==model::TensorRole::expert_up);
    CHECK(model::infer_tensor_role("blk.0.ffn_down_exps.weight")==model::TensorRole::expert_down);
    {
        std::vector<http::ChatMessage> messages{{"system","be concise"},{"user","hello"}};
        std::string prompt,format_error;
        CHECK(chat::format_legacy_chat("{% legacy chatml %}","qwen2",messages,prompt,&format_error));
        CHECK(prompt.find("<|im_start|>system")!=std::string::npos);
        CHECK(prompt.find("<|im_start|>assistant\n")!=std::string::npos);
        std::vector<http::ChatMessage> bad{{"tool","result"}};prompt.clear();
        CHECK(!chat::format_legacy_chat("","llama",bad,prompt,&format_error));
    }
    planner::ExecutionPlanner planner;auto prefill_plan=planner.select(planner::Phase::prefill,true,8,512,1024);auto decode_plan=planner.select(planner::Phase::decode,true,1,512,1024);
    CHECK(prefill_plan.kernel==planner::Kernel::quantized_gemm);CHECK(decode_plan.kernel==planner::Kernel::quantized_gemv);CHECK(prefill_plan.workers>=1&&decode_plan.workers>=1);
    CHECK(quant::find("Q1_SIGNED") && quant::find("Q18_SIGNED"));
    CHECK(quant::find("F16")->implemented);
    // ZI-012 GREEN: every GGML storage format must now be executable, not
    // merely recognised by name.
    CHECK(quant::find("Q2_K")->implemented);
    CHECK(quant::find("Q4_K")->implemented);
    CHECK(quant::find("Q5_K")->implemented);
    CHECK(quant::find("Q8_K")->implemented);
    CHECK(quant::find("IQ1_S")->implemented);
    CHECK(quant::find("IQ1_M")->implemented);
    CHECK(quant::find("IQ2_XXS")->implemented);
    CHECK(quant::find("IQ2_XS")->implemented);
    CHECK(quant::find("IQ2_S")->implemented);
    CHECK(quant::find("IQ3_XXS")->implemented);
    CHECK(quant::find("IQ3_S")->implemented);
    CHECK(quant::find("IQ4_XS")->implemented);
    CHECK(quant::find("MXFP4")->implemented);
    CHECK(quant::find("Q5_0")->implemented && quant::find("Q3_K")->implemented);
    CHECK(quant::find("Q6_K")->implemented && quant::find("IQ4_NL")->implemented);
    // ZI-012 RED: formats must be executable, not merely recognised by name.
    CHECK(quant::find("Q4_1")->implemented);
    CHECK(quant::find("Q5_1")->implemented);
    CHECK(quant::find("BF16")->implemented);
    {
        std::array<std::uint8_t,20> block{}; // Q4_1: d=1, m=2, all q=0.
        block[0]=0x00; block[1]=0x3c; block[2]=0x00; block[3]=0x40;
        std::array<float,32> out{};
        CHECK(quant::dequantize(*quant::find("Q4_1"),block.data(),32,out.data()));
        for(float v:out) CHECK(v==2.0f);
    }
    {
        std::array<std::uint8_t,24> block{}; // Q5_1: d=1, m=2, all q=0.
        block[0]=0x00; block[1]=0x3c; block[2]=0x00; block[3]=0x40;
        std::array<float,32> out{};
        CHECK(quant::dequantize(*quant::find("Q5_1"),block.data(),32,out.data()));
        for(float v:out) CHECK(v==2.0f);
    }
    {
        const std::uint16_t bf16[]{0x3f80,0xc000,0x0001,0x7f80};
        std::array<std::uint8_t,8> bytes{}; for(std::size_t i=0;i<4;++i){bytes[i*2]=std::uint8_t(bf16[i]);bytes[i*2+1]=std::uint8_t(bf16[i]>>8);}
        std::array<float,4> out{};
        CHECK(quant::dequantize(*quant::find("BF16"),bytes.data(),4,out.data()));
        CHECK(out[0]==1.0f && out[1]==-2.0f && out[2]>0.0f && std::isinf(out[3]));
        CHECK(quant::row_bytes(*quant::find("BF16"),4)==8);
    }
    CHECK(quant::row_bytes(*quant::find("F32"),std::numeric_limits<std::size_t>::max())==0);

    // ZI-012: K-quant, IQ and MXFP4 super-block byte layouts must match GGML.
    CHECK(quant::row_bytes(*quant::find("Q2_K"),256)==84);
    CHECK(quant::row_bytes(*quant::find("Q4_K"),256)==144);
    CHECK(quant::row_bytes(*quant::find("Q5_K"),256)==176);
    CHECK(quant::row_bytes(*quant::find("Q8_K"),256)==292);
    CHECK(quant::row_bytes(*quant::find("IQ1_S"),256)==50);
    CHECK(quant::row_bytes(*quant::find("IQ1_M"),256)==56);
    CHECK(quant::row_bytes(*quant::find("IQ2_XXS"),256)==66);
    CHECK(quant::row_bytes(*quant::find("IQ2_XS"),256)==74);
    CHECK(quant::row_bytes(*quant::find("IQ2_S"),256)==82);
    CHECK(quant::row_bytes(*quant::find("IQ3_XXS"),256)==98);
    CHECK(quant::row_bytes(*quant::find("IQ3_S"),256)==110);
    CHECK(quant::row_bytes(*quant::find("IQ4_XS"),256)==136);
    CHECK(quant::row_bytes(*quant::find("MXFP4"),32)==17);

    // Q2_K exact block: d=1, dmin=0, every 6-bit scale = 1, 2-bit pattern 0..3.
    {
        std::array<std::uint8_t,84> q2k{};
        for(std::size_t i=0;i<16;++i)q2k[i]=0x01;
        for(std::size_t i=16;i<80;++i)q2k[i]=0xE4;
        q2k[80]=0x00;q2k[81]=0x3c;q2k[82]=0x00;q2k[83]=0x00;
        std::array<float,256> out{};
        CHECK(quant::dequantize(*quant::find("Q2_K"),q2k.data(),256,out.data()));
        for(std::size_t chunk=0;chunk<2;++chunk)
            for(std::size_t i=0;i<128;++i)CHECK(out[chunk*128+i]==static_cast<float>(i/32));
    }
    // Q4_K exact block: d=1, dmin=0, all scales 1, low nibbles 1 / high nibbles 2.
    {
        std::array<std::uint8_t,144> q4k{};
        q4k[0]=0x00;q4k[1]=0x3c;q4k[2]=0x00;q4k[3]=0x00;
        q4k[4]=0x01;q4k[5]=0x01;q4k[6]=0x01;q4k[7]=0x01;
        q4k[12]=0x01;q4k[13]=0x01;q4k[14]=0x01;q4k[15]=0x01;
        for(std::size_t i=16;i<144;++i)q4k[i]=0x21;
        std::array<float,256> out{};
        CHECK(quant::dequantize(*quant::find("Q4_K"),q4k.data(),256,out.data()));
        for(std::size_t i=0;i<256;++i)CHECK(out[i]==((i/32)%2==0?1.0f:2.0f));
    }
    // Q5_K: Q4_K packing plus a high-bit plane; qh[0] bit 0 lifts output 0 by 16.
    {
        std::array<std::uint8_t,176> q5k{};
        q5k[0]=0x00;q5k[1]=0x3c;q5k[2]=0x00;q5k[3]=0x00;
        q5k[4]=0x01;q5k[5]=0x01;q5k[6]=0x01;q5k[7]=0x01;
        q5k[12]=0x01;q5k[13]=0x01;q5k[14]=0x01;q5k[15]=0x01;
        q5k[16]=0x01;
        for(std::size_t i=48;i<176;++i)q5k[i]=0x21;
        std::array<float,256> out{};
        CHECK(quant::dequantize(*quant::find("Q5_K"),q5k.data(),256,out.data()));
        for(std::size_t i=0;i<256;++i){
            const float base=(i/32)%2==0?1.0f:2.0f;
            CHECK(out[i]==(i==0?base+16.0f:base));
        }
    }
    // Q8_K exact layout: f32 delta = 1.0, quants ramp from -128 as int8.
    {
        std::vector<std::uint8_t> q8k(292);
        q8k[0]=0x00;q8k[1]=0x00;q8k[2]=0x80;q8k[3]=0x3f;
        for(std::size_t i=4;i<292;++i)q8k[i]=static_cast<std::uint8_t>(i-4);
        std::vector<float> out(256);
        CHECK(quant::dequantize(*quant::find("Q8_K"),q8k.data(),256,out.data()));
        for(std::size_t i=0;i<256;++i)CHECK(out[i]==(i<128?static_cast<float>(i):static_cast<float>(i)-256.0f));
    }
    // MXFP4 exact block: E8M0 scale for 0.0625 (e=124 -> 2^-4). Byte qs[0]
    // = 0x78: low nibble 8 (code 8 = -0 = 0) decodes out[0], high nibble 7
    // (code 7 = 12, the doubled 6.0) decodes out[16] per kvalues_fp4.
    {
        std::array<std::uint8_t,17> mx{};
        mx[0]=124;      // E8M0: GGML_E8M0_TO_FP32_HALF(124) = 2^-4 = 0.0625
        mx[1]=0x78;     // lo nibble 8 (0) -> out[0]; hi nibble 7 (12) -> out[16]
        std::array<float,32> out{};
        CHECK(quant::dequantize(*quant::find("MXFP4"),mx.data(),32,out.data()));
        CHECK(out[0]==0.0f);
        CHECK(out[16]==12.0f*0.0625f);
        for(std::size_t i=1;i<16;++i)CHECK(out[i]==0.0f);
        for(std::size_t i=17;i<32;++i)CHECK(out[i]==0.0f);
    }
    // GEMV coverage: the quantised dot must equal a float64 dot of decoded values
    // for one newly supported super-block format of each family.
    auto check_super_gemv=[&](const char* name,std::size_t bytes){
        std::vector<std::uint8_t> block(bytes);
        for(std::size_t i=0;i<bytes;++i)block[i]=static_cast<std::uint8_t>(i*29+7);
        std::vector<float> input(256),decoded(256);
        for(std::size_t i=0;i<256;++i)input[i]=float(int(i%17)-8)*0.03125f;
        float actual=0;
        CHECK(quant::gemv(*quant::find(name),block.data(),input.data(),&actual,1,256));
        CHECK(quant::dequantize(*quant::find(name),block.data(),256,decoded.data()));
        double reference=0;for(std::size_t i=0;i<256;++i)reference+=double(decoded[i])*input[i];
        CHECK(std::abs(actual-static_cast<float>(reference))<1.0e-3f*std::max(1.0f,std::abs(static_cast<float>(reference))));
    };
    check_super_gemv("Q2_K",84);
    check_super_gemv("Q4_K",144);
    check_super_gemv("Q5_K",176);
    check_super_gemv("Q8_K",292);
    check_super_gemv("IQ4_XS",136);
    check_super_gemv("IQ3_XXS",98);
    check_super_gemv("IQ2_XXS",66);
    check_super_gemv("IQ1_S",50);
    check_super_gemv("TQ2_0",66);


    // Generic signed bit-packed Q2: values [-2,-1,0,1], packed little-endian.
    const std::uint8_t q2[]{0x4e}; float decoded[4]{};
    CHECK(quant::dequantize(*quant::find("Q2_SIGNED"),q2,4,decoded,0.5f));
    CHECK(decoded[0]==-1.0f && decoded[1]==-0.5f && decoded[2]==0.0f && decoded[3]==0.5f);

    // Ternary code sequence: -1, 0, +1, -1.
    const std::uint8_t tr[]{0x24}; float tern[4]{};
    CHECK(quant::dequantize(*quant::find("TERNARY2"),tr,4,tern,2.0f));
    CHECK(tern[0]==-2.0f && tern[1]==0.0f && tern[2]==2.0f && tern[3]==-2.0f);

    // IEEE half tests: 1.0, -2.0, 0.5, +infinity.
    const std::uint16_t h[]{0x3c00,0xc000,0x3800,0x7c00};float f[4]{};
    CHECK(quant::decode_fp16(h,f,4));
    CHECK(f[0]==1.0f && f[1]==-2.0f && f[2]==0.5f && std::isinf(f[3]));
    // Every non-NaN binary16 value must survive binary16 -> float32 -> binary16.
    for (std::uint32_t bits=0; bits<=0xffffu; ++bits) {
        const std::uint16_t hbits=static_cast<std::uint16_t>(bits);
        float decoded{}; std::uint16_t encoded{};
        CHECK(quant::decode_fp16(&hbits,&decoded,1));
        CHECK(quant::encode_fp16(&decoded,&encoded,1));
        if ((hbits & 0x7c00u) != 0x7c00u || (hbits & 0x03ffu) == 0)
            CHECK(encoded==hbits);
        else
            CHECK((encoded & 0x7c00u)==0x7c00u && (encoded & 0x03ffu)!=0);
    }
    // Round-to-nearest, ties-to-even around 1.0 and the subnormal boundary.
    const float round_inputs[]{1.0f,1.00048828125f,1.00146484375f,
        5.960464477539063e-8f,2.9802322387695312e-8f,
        65520.0f,-0.0f,std::numeric_limits<float>::infinity()};
    const std::uint16_t round_expected[]{0x3c00,0x3c00,0x3c02,0x0001,0x0000,0x7c00,0x8000,0x7c00};
    std::uint16_t round_actual[8]{};
    CHECK(quant::encode_fp16(round_inputs,round_actual,8));
    for (std::size_t i=0;i<8;++i) CHECK(round_actual[i]==round_expected[i]);

    // Q8_0 one-row GEMV: fp16 scale 1.0 and 32 unit signed weights.
    std::array<std::uint8_t,34> q8{}; q8[0]=0x00; q8[1]=0x3c;
    for(std::size_t i=2;i<q8.size();++i)q8[i]=1;
    std::array<float,32> ones{}; ones.fill(1.0f); float dot{};
    CHECK(quant::gemv(*quant::find("Q8_0"),q8.data(),ones.data(),&dot,1,32));
    CHECK(dot==32.0f);

    // Q8_0 fused GEMV is compared with the scalar dequantise-then-dot reference.
    std::vector<std::uint8_t> q8matrix(2*68);
    std::vector<float> q8input(64), q8actual(2), q8row(64);
    for(std::size_t i=0;i<64;++i)q8input[i]=float(int(i%13)-6)*0.125f;
    for(std::size_t r=0;r<2;++r)for(std::size_t b=0;b<2;++b){
        const std::size_t off=r*68+b*34;q8matrix[off]=0x00;q8matrix[off+1]=0x34; // fp16 0.25
        for(std::size_t i=0;i<32;++i)q8matrix[off+2+i]=static_cast<std::uint8_t>(static_cast<std::int8_t>(int(((b*32+i)*7+r*11)%31)-15));
    }
    CHECK(quant::gemv(*quant::find("Q8_0"),q8matrix.data(),q8input.data(),q8actual.data(),2,64));
    for(std::size_t r=0;r<2;++r){
        CHECK(quant::dequantize(*quant::find("Q8_0"),q8matrix.data()+r*68,64,q8row.data()));
        double ref=0;for(std::size_t i=0;i<64;++i)ref+=double(q8row[i])*q8input[i];
        CHECK(std::abs(q8actual[r]-static_cast<float>(ref))<2.0e-4f);
    }

    // Exact GGML block decoders: zero-valued blocks and IQ4_NL lookup ordering.
    std::array<std::uint8_t,22> q5{}; q5[0]=0x00; q5[1]=0x3c;
    for(std::size_t i=6;i<22;++i)q5[i]=0x88;
    std::array<float,32> q5out{}; CHECK(quant::dequantize(*quant::find("Q5_0"),q5.data(),32,q5out.data()));
    for(float x:q5out)CHECK(x==-8.0f);
    std::array<std::uint8_t,110> q3{}; q3[108]=0x00; q3[109]=0x3c; for(std::size_t i=0;i<16;++i)q3[i]=1;
    std::array<float,256> q3out{};
    CHECK(quant::dequantize(*quant::find("Q3_K"),q3.data(),256,q3out.data()));
    for(std::size_t i=0;i<16;++i)CHECK(q3out[i]==0.0f);
    for(std::size_t i=16;i<256;++i)CHECK(q3out[i]==128.0f);
    std::array<std::uint8_t,210> q6{}; q6[208]=0x00; q6[209]=0x3c;
    for(std::size_t i=192;i<208;++i)q6[i]=1; q6[0]=1;
    std::array<float,256> q6out{};
    CHECK(quant::dequantize(*quant::find("Q6_K"),q6.data(),256,q6out.data()));
    CHECK(q6out[0]==-31.0f && q6out[32]==-32.0f && q6out[64]==-32.0f && q6out[96]==-32.0f);
    for(std::size_t i=1;i<256;++i)if(i!=32&&i!=64&&i!=96)CHECK(q6out[i]==-32.0f);
    std::array<std::uint8_t,18> iq4{}; iq4[0]=0x00; iq4[1]=0x3c; for(std::size_t i=2;i<18;++i)iq4[i]=0x88;
    std::array<float,32> iq4out{}; CHECK(quant::dequantize(*quant::find("IQ4_NL"),iq4.data(),32,iq4out.data()));
    for(float x:iq4out)CHECK(x==1.0f);

    // Official GGML BitNet ternary blocks: TQ1_0 base-3 packing and TQ2_0
    // 2-bit packing. Per ggml-common.h the f16 delta sits at the END of the
    // block: block_tq1_0={qs[48];qh[4];d} (d at offset 52), block_tq2_0=
    // {qs[64];d} (d at offset 64).
    std::array<std::uint8_t,54> tq1{};tq1[52]=0x00;tq1[53]=0x3c;for(std::size_t i=0;i<52;++i)tq1[i]=0xff;
    std::array<float,256> tq1out{};CHECK(quant::dequantize(*quant::find("TQ1_0"),tq1.data(),256,tq1out.data()));for(float x:tq1out)CHECK(x==1.0f);
    std::array<std::uint8_t,54> tq1zero{};tq1zero[52]=0x00;tq1zero[53]=0x3c;
    CHECK(quant::dequantize(*quant::find("TQ1_0"),tq1zero.data(),256,tq1out.data()));for(float x:tq1out)CHECK(x==-1.0f);
    std::array<std::uint8_t,66> tq2{};tq2[64]=0x00;tq2[65]=0x3c;for(std::size_t i=0;i<64;++i)tq2[i]=0xe4;
    std::array<float,256> tq2out{};CHECK(quant::dequantize(*quant::find("TQ2_0"),tq2.data(),256,tq2out.data()));
    for(std::size_t i=0;i<256;++i){const float expected=float(int((i/32)%4)-1);CHECK(tq2out[i]==expected);}
    CHECK(quant::row_bytes(*quant::find("TQ1_0"),256)==54);CHECK(quant::row_bytes(*quant::find("TQ2_0"),256)==66);

    // Streaming block-GEMV coverage for each runtime quant format other than the Q8_0 fast path.
    auto check_quant_gemv=[&](const quant::Format& fmt,std::vector<std::uint8_t> packed,std::size_t n){
        std::vector<float> input(n),decoded(n);for(std::size_t i=0;i<n;++i)input[i]=float(int(i%17)-8)*0.03125f;
        float actual=0;CHECK(quant::gemv(fmt,packed.data(),input.data(),&actual,1,n));CHECK(quant::dequantize(fmt,packed.data(),n,decoded.data()));
        double reference=0;for(std::size_t i=0;i<n;++i)reference+=double(decoded[i])*input[i];CHECK(std::abs(actual-static_cast<float>(reference))<1.0e-3f*std::max(1.0f,std::abs(static_cast<float>(reference))));
    };
    std::vector<std::uint8_t> q5block(22);q5block[0]=0x00;q5block[1]=0x3c;for(std::size_t i=2;i<q5block.size();++i)q5block[i]=static_cast<std::uint8_t>(i*13);check_quant_gemv(*quant::find("Q5_0"),q5block,32);
    std::vector<std::uint8_t> q3block(110);for(std::size_t i=0;i<108;++i)q3block[i]=static_cast<std::uint8_t>(i*7);q3block[108]=0x00;q3block[109]=0x3c;check_quant_gemv(*quant::find("Q3_K"),q3block,256);
    std::vector<std::uint8_t> q6block(210);for(std::size_t i=0;i<208;++i)q6block[i]=static_cast<std::uint8_t>(i*11);q6block[208]=0x00;q6block[209]=0x3c;check_quant_gemv(*quant::find("Q6_K"),q6block,256);
    std::vector<std::uint8_t> iq4block(18);iq4block[0]=0x00;iq4block[1]=0x3c;for(std::size_t i=2;i<18;++i)iq4block[i]=static_cast<std::uint8_t>(i*9);check_quant_gemv(*quant::find("IQ4_NL"),iq4block,32);
    // Worker-count invariance and the parallel row dispatch path.
    const auto* q8fmt=quant::find("Q8_0");std::vector<std::uint8_t> q8parallel_matrix(128*34);for(std::size_t i=0;i<128;++i){q8parallel_matrix[i*34]=0x00;q8parallel_matrix[i*34+1]=0x3c;for(std::size_t j=2;j<34;++j)q8parallel_matrix[i*34+j]=static_cast<std::uint8_t>((i*7+j*3)&255);}
    std::vector<float> q8parallel_input(32),q8single(128),q8parallel(128);for(std::size_t i=0;i<32;++i)q8parallel_input[i]=float(int(i%9)-4)*0.1f;
    CHECK(quant::gemv(*q8fmt,q8parallel_matrix.data(),q8parallel_input.data(),q8single.data(),128,32,1.0f,1));CHECK(quant::gemv(*q8fmt,q8parallel_matrix.data(),q8parallel_input.data(),q8parallel.data(),128,32,1.0f,4));for(std::size_t i=0;i<128;++i)CHECK(std::abs(q8single[i]-q8parallel[i])<1.0e-4f);
    std::vector<float> batch_input(3*32),batch_single(3*128),batch_parallel(3*128);for(std::size_t i=0;i<batch_input.size();++i)batch_input[i]=float(int(i%13)-6)*0.03f;CHECK(quant::gemm(*q8fmt,q8parallel_matrix.data(),batch_input.data(),batch_single.data(),3,128,32,1.0f,1));CHECK(quant::gemm(*q8fmt,q8parallel_matrix.data(),batch_input.data(),batch_parallel.data(),3,128,32,1.0f,4));for(std::size_t i=0;i<batch_single.size();++i)CHECK(std::abs(batch_single[i]-batch_parallel[i])<1.0e-4f);

    // SentencePiece byte-level BPE: merges are selected by token score and decode reversibly.
    model::ModelConfig tc;tc.metadata["tokenizer.ggml.model"]="llama";tc.bos_token_id=1;tc.eos_token_id=2;tc.unknown_token_id=0;tc.add_space_prefix=true;
    tc.tokenizer_tokens={"<unk>","<s>","</s>","a","b","\xE2\x96\x81" "a","\xE2\x96\x81" "b"};
    tc.tokenizer_scores={-1000,-1000,-1000,0,0,5,5};tc.tokenizer_token_types={3,3,3,1,1,1,1};
    tokenizer::ModelTokenizer tok;std::string tok_error;CHECK(tok.load(tc,&tok_error));
    std::vector<std::uint32_t> ids;CHECK(tok.encode("a b",ids,true,false));
    CHECK(ids.size()==3&&ids[0]==1&&ids[1]==5&&ids[2]==6);
    std::string decoded_text;CHECK(tok.decode(ids,decoded_text));CHECK(decoded_text=="a b");
    // GPT-2/Qwen byte-level BPE: byte symbols are merged by token rank and decoded back to bytes.
    model::ModelConfig gc;gc.metadata["tokenizer.ggml.model"]="gpt2";gc.bos_token_id=1;gc.eos_token_id=2;gc.unknown_token_id=0;gc.add_space_prefix=false;
    gc.tokenizer_tokens={"<unk>","<s>","</s>","a","b","ab","\xC4\xA0","\xC4\xA0" "b","a" "\xC4\xA0" "b"};gc.tokenizer_scores={0,0,0,0,0,0,0,0,0};gc.tokenizer_token_types={3,3,3,1,1,1,1,1,1};
    tokenizer::ModelTokenizer gpt;CHECK(gpt.load(gc,&tok_error));std::vector<std::uint32_t> gpt_ids;CHECK(gpt.encode("ab",gpt_ids,false,false));CHECK(gpt_ids.size()==1&&gpt_ids[0]==5);std::string gpt_text;CHECK(gpt.decode(gpt_ids,gpt_text));CHECK(gpt_text=="ab");
    CHECK(gpt.encode("a b",gpt_ids,false,false));CHECK(gpt_ids.size()==2&&gpt_ids[0]==3&&gpt_ids[1]==7);CHECK(gpt.decode(gpt_ids,gpt_text));CHECK(gpt_text=="a b");
    // Unicode GPT-2 pre-tokenisation: a Unicode letter and following punctuation
    // must remain separate even when the synthetic vocabulary contains a cross-boundary token.
    gc.tokenizer_tokens.push_back("\xC3\x83");       // byte 0xC3 -> U+00C3
    gc.tokenizer_tokens.push_back("\xC2\xA9");       // byte 0xA9 -> U+00A9
    gc.tokenizer_tokens.push_back("\xC3\x83\xC2\xA9"); // encoded UTF-8 for source byte pair C3 A9
    gc.tokenizer_tokens.push_back("!");
    gc.tokenizer_tokens.push_back("\xC3\x83\xC2\xA9!");
    gc.tokenizer_scores.assign(gc.tokenizer_tokens.size(),0.0f);
    gc.tokenizer_token_types.assign(gc.tokenizer_tokens.size(),1);
    CHECK(gpt.load(gc,&tok_error));
    CHECK(gpt.encode("\xC3\xA9!",gpt_ids,false,false));
    CHECK(gpt_ids.size()==2&&gpt_ids[0]==11&&gpt_ids[1]==12);

    // GGUF invalid-magic rejection is part of the parser safety gate.
    const auto tmp=std::filesystem::temp_directory_path()/"zeninfer-invalid.gguf";
    {std::ofstream o(tmp,std::ios::binary);o<<"NOPE";}
    auto report=gguf::inspect(tmp);CHECK(!report.ok);CHECK(report.error.find("magic")!=std::string::npos);
    std::filesystem::remove(tmp);

    // Minimal valid GGUF v3 with one string metadata entry and no tensors.
    const auto valid=std::filesystem::temp_directory_path()/"zeninfer-minimal.gguf";
    auto w32=[](std::ofstream&o,std::uint32_t v){o.write(reinterpret_cast<const char*>(&v),4);};
    auto w64=[](std::ofstream&o,std::uint64_t v){o.write(reinterpret_cast<const char*>(&v),8);};
    auto ws=[&](std::ofstream&o,const std::string&v){w64(o,v.size());o.write(v.data(),static_cast<std::streamsize>(v.size()));};
    {std::ofstream o(valid,std::ios::binary);o.write("GGUF",4);w32(o,3);w64(o,0);w64(o,1);ws(o,"general.architecture");w32(o,8);ws(o,"qwen3");const auto pos=static_cast<std::uint64_t>(o.tellp());for(std::uint64_t i=pos;i<((pos+31)&~std::uint64_t(31));++i)o.put('\0');}
    auto good=gguf::inspect(valid);CHECK(good.ok);CHECK(good.config.architecture==model::Architecture::qwen);
    CHECK(good.tensor_data_offset>=4+4+8+8);
    std::filesystem::remove(valid);

    // GGUF Boolean metadata is a one-byte value. IQ4_NL uses 32 values / 18 bytes.
    const auto typed=std::filesystem::temp_directory_path()/"zeninfer-bool-iq4nl.gguf";
    {std::ofstream o(typed,std::ios::binary);o.write("GGUF",4);w32(o,3);w64(o,1);w64(o,2);
     ws(o,"general.architecture");w32(o,8);ws(o,"llama");
     ws(o,"tokenizer.ggml.add_bos_token");w32(o,7);const std::uint8_t yes=1;o.write(reinterpret_cast<const char*>(&yes),1);
     ws(o,"test.weight");w32(o,1);w64(o,32);w32(o,20);w64(o,0);
     const auto pos=static_cast<std::uint64_t>(o.tellp());for(std::uint64_t i=pos;i<((pos+31)&~std::uint64_t(31));++i)o.put('\0');
     std::array<std::uint8_t,18> payload{};o.write(reinterpret_cast<const char*>(payload.data()),payload.size());}
    auto typed_report=gguf::inspect(typed);CHECK(typed_report.ok);CHECK(typed_report.tensors.size()==1);
    if(typed_report.ok&&!typed_report.tensors.empty())CHECK(typed_report.tensors[0].byte_size==18);
    std::filesystem::remove(typed);

    // Canonical GGML type-ID table: IQ formats and current scalar/low-bit types
    // must be size-validated, while removed legacy IDs must not be guessed.
    auto check_ggml_type=[&](std::uint32_t type,std::uint64_t elements,std::uint64_t expected){
        const auto file=std::filesystem::temp_directory_path()/("zeninfer-type-"+std::to_string(type)+".gguf");
        {std::ofstream o(file,std::ios::binary);o.write("GGUF",4);w32(o,3);w64(o,1);w64(o,1);ws(o,"general.architecture");w32(o,8);ws(o,"llama");ws(o,"weight");w32(o,1);w64(o,elements);w32(o,type);w64(o,0);const auto pos=static_cast<std::uint64_t>(o.tellp());for(std::uint64_t i=pos;i<((pos+31)&~std::uint64_t(31));++i)o.put('\0');if(expected){std::vector<char> payload(static_cast<std::size_t>(expected));o.write(payload.data(),static_cast<std::streamsize>(payload.size()));}}
        const auto result=gguf::inspect(file);if(expected){CHECK(result.ok);if(result.ok)CHECK(result.tensors[0].byte_size==expected);}else CHECK(!result.ok);
        std::filesystem::remove(file);
    };
    check_ggml_type(16,256,66);check_ggml_type(17,256,74);check_ggml_type(18,256,98);check_ggml_type(19,256,50);check_ggml_type(20,32,18);
    check_ggml_type(21,256,110);check_ggml_type(22,256,82);check_ggml_type(23,256,136);
    check_ggml_type(24,1,1);check_ggml_type(25,1,2);check_ggml_type(26,1,4);check_ggml_type(27,1,8);check_ggml_type(28,1,8);
    check_ggml_type(29,256,56);check_ggml_type(30,1,2);check_ggml_type(34,256,54);check_ggml_type(35,256,66);
    check_ggml_type(39,32,17);
    for(auto removed:{4u,5u,31u,32u,33u,36u,37u,38u,40u,41u,42u})check_ggml_type(removed,256,0);
    // End-to-end GGUF tensor read/decode for a BitNet TQ2_0 block (f16 delta
    // at the end of the 66-byte block, per block_tq2_0={qs[64];d}).
    const auto tqfile=std::filesystem::temp_directory_path()/"zeninfer-bitnet-tq2.gguf";
    {std::ofstream o(tqfile,std::ios::binary);o.write("GGUF",4);w32(o,3);w64(o,1);w64(o,1);ws(o,"general.architecture");w32(o,8);ws(o,"bitnet");ws(o,"token_embd.weight");w32(o,1);w64(o,256);w32(o,35);w64(o,0);const auto pos=static_cast<std::uint64_t>(o.tellp());for(std::uint64_t i=pos;i<((pos+31)&~std::uint64_t(31));++i)o.put('\0');std::array<std::uint8_t,66> block{};block[64]=0x00;block[65]=0x3c;for(std::size_t i=0;i<64;++i)block[i]=0xe4;o.write(reinterpret_cast<const char*>(block.data()),block.size());}
    const auto tqreport=gguf::inspect(tqfile);CHECK(tqreport.ok);std::vector<float> tqdecoded;std::string tqerror;CHECK(gguf::read_tensor_f32(tqfile,tqreport,"token_embd.weight",tqdecoded,&tqerror));CHECK(tqdecoded.size()==256);if(tqdecoded.size()==256)for(std::size_t i=0;i<256;++i)CHECK(tqdecoded[i]==float(int((i/32)%4)-1));std::filesystem::remove(tqfile);
    // Persistent compute pool: every partition is visited exactly once and the
    // caller can request a single-thread deterministic execution path.
    std::vector<std::atomic<unsigned>> visits(257);for(auto& v:visits)v.store(0);
    zen::ComputePool::instance().parallel_for(visits.size(),4,[&](std::size_t first,std::size_t last){for(std::size_t i=first;i<last;++i)visits[i].fetch_add(1);});
    for(const auto& v:visits)CHECK(v.load()==1);
    std::uint64_t pool_sum=0;zen::ComputePool::instance().parallel_for(100,1,[&](std::size_t first,std::size_t last){for(std::size_t i=first;i<last;++i)pool_sum+=i;});
    CHECK(pool_sum==4950);
    // Concurrent callers must not lose, duplicate or cross-contaminate jobs.
    std::array<std::atomic<unsigned>,4> concurrent_visits{};
    std::array<std::thread,4> pool_callers;
    for(std::size_t caller=0;caller<pool_callers.size();++caller) pool_callers[caller]=std::thread([&,caller]{
        zen::ComputePool::instance().parallel_for(1000,4,[&,caller](std::size_t first,std::size_t last){
            for(std::size_t i=first;i<last;++i) concurrent_visits[caller].fetch_add(1,std::memory_order_relaxed);
        });
    });
    for(auto& t:pool_callers)t.join();
    for(const auto& count:concurrent_visits)CHECK(count.load()==1000);
    // Exceptions raised by worker partitions must reach the invoking caller.
    bool pool_exception_seen=false;
    try { zen::ComputePool::instance().parallel_for(64,4,[](std::size_t first,std::size_t){if(first!=0)throw std::runtime_error("worker failure");}); }
    catch(const std::runtime_error&) { pool_exception_seen=true; }
    CHECK(pool_exception_seen);

    // ALiBi slopes decrease by head index and apply a non-positive causal bias.
    CHECK(zen::attention::alibi_head_slope(0,8)>zen::attention::alibi_head_slope(1,8));
    CHECK(zen::attention::alibi_head_slope(0,8)>0.0f);
    CHECK(zen::attention::alibi_bias(0,8,10,10)==0.0f);
    CHECK(zen::attention::alibi_bias(0,8,10,9)<0.0f);
    CHECK(std::isinf(zen::attention::alibi_bias(0,8,9,10)));
    // Independent ALiBi reference values for the power-of-two construction.
    const std::array<float,8> alibi_reference={0.5f,0.25f,0.125f,0.0625f,0.03125f,0.015625f,0.0078125f,0.00390625f};
    for(std::uint32_t h=0;h<alibi_reference.size();++h) {
        CHECK(std::fabs(zen::attention::alibi_head_slope(h,8)-alibi_reference[h])<1e-7f);
        for(std::uint32_t distance=0;distance<16;++distance) {
            const float expected=-alibi_reference[h]*static_cast<float>(distance);
            CHECK(std::fabs(zen::attention::alibi_bias(h,8,32,32-distance)-expected)<1e-6f);
        }
    }
    CHECK(zen::attention::alibi_head_slope(8,8)==0.0f);
    CHECK(zen::attention::alibi_head_slope(0,0)==0.0f);

    // Differential binary16 corpus generated independently with Python struct.pack('<e').
    for(const auto& sample:fp16_reference::corpus) {
        float input_value{};std::memcpy(&input_value,&sample.f32,sizeof(input_value));
        std::uint16_t actual{};
        CHECK(quant::encode_fp16(&input_value,&actual,1));
        CHECK(actual==sample.f16);
    }

    // ZI-016: GGUF security hardening regressions. All three cases crashed or
    // corrupted state before the hardening port; every one must now fail closed
    // with an error string, and the file and memory parse surfaces must agree.
    {
        // 1) Metadata ARRAY recursion cap: 80 nested one-element arrays exceed
        //    kMaxMetaDepth=64. A 2 MB file with 180k levels crashed the parser
        //    with 0xC00000FD before the cap existed.
        const auto deep=std::filesystem::temp_directory_path()/"zeninfer-deep-array.gguf";
        std::ofstream o(deep,std::ios::binary);
        o.write("GGUF",4);w32(o,3);w64(o,0);w64(o,1);
        ws(o,"crafted.nested");w32(o,9);
        for(unsigned level=0;level<80;++level){w32(o,9);w64(o,1);}
        w32(o,8);w64(o,0);o.close();
        auto deep_report=gguf::inspect(deep);CHECK(!deep_report.ok);
        const auto raw=std::filesystem::path(deep);
        std::vector<char> image(static_cast<std::size_t>(std::filesystem::file_size(raw)));
        {std::ifstream in(raw,std::ios::binary);in.read(image.data(),static_cast<std::streamsize>(image.size()));}
        auto mem_report=gguf::inspect_bytes(reinterpret_cast<const std::uint8_t*>(image.data()),image.size());
        CHECK(!mem_report.ok);
        // The file and memory surfaces must agree on every input.
        CHECK(mem_report.error==deep_report.error);
        std::filesystem::remove(deep);
    }
    {
        // 2) u64-width gate: llama.embedding_length = 2^32+4096 must not be
        //    applied to the config. The file stays structurally valid, but the
        //    value is rejected instead of silently truncating to 4096 (which
        //    previously shaped the model config from hostile input).
        auto w64le=[](std::vector<std::uint8_t>& v,std::uint64_t x){for(int i=0;i<8;++i){v.push_back(static_cast<std::uint8_t>(x));x>>=8;}};
        auto w32le=[](std::vector<std::uint8_t>& v,std::uint32_t x){for(int i=0;i<4;++i){v.push_back(static_cast<std::uint8_t>(x));x>>=8;}};
        std::vector<std::uint8_t> wide{'G','G','U','F',3,0,0,0};
        w64le(wide,0);w64le(wide,1);
        const char* key="llama.embedding_length";
        w64le(wide,22);wide.insert(wide.end(),key,key+22);
        w32le(wide,10); // U64
        w64le(wide,0x100000000ull+4096ull);
        while(wide.size()%32)wide.push_back(0); // align the tensor-data section
        auto wide_report=gguf::inspect_bytes(wide.data(),wide.size());
        CHECK(wide_report.ok);                                  // structurally valid image
        if(wide_report.ok)CHECK(wide_report.config.embedding_length==0); // value not applied
    }
    {
        // 3) Hostile buffer: a metadata key whose declared string length runs
        //    past the end of the buffer must fail closed, and a null/empty
        //    buffer must be rejected without dereferencing anything.
        auto w64le=[](std::vector<std::uint8_t>& v,std::uint64_t x){for(int i=0;i<8;++i){v.push_back(static_cast<std::uint8_t>(x));x>>=8;}};
        auto w32le=[](std::vector<std::uint8_t>& v,std::uint32_t x){for(int i=0;i<4;++i){v.push_back(static_cast<std::uint8_t>(x));x>>=8;}};
        std::vector<std::uint8_t> hostile{'G','G','U','F',3,0,0,0};
        w64le(hostile,0);w64le(hostile,1);w64le(hostile,1ull<<24);
        auto hostile_report=gguf::inspect_bytes(hostile.data(),hostile.size());
        CHECK(!hostile_report.ok);
        auto null_report=gguf::inspect_bytes(nullptr,0);
        CHECK(!null_report.ok);
        // Truncated header: fewer bytes than the GGUF magic itself.
        const std::uint8_t stub[]={'G','G'};
        auto stub_report=gguf::inspect_bytes(stub,sizeof(stub));
        CHECK(!stub_report.ok);
        // A valid minimal image parsed from memory must agree with the file path.
        std::vector<std::uint8_t> minimal{'G','G','U','F',3,0,0,0};
        w64le(minimal,0);w64le(minimal,1); // zero tensors, one metadata entry
        const char* arch="general.architecture"; // 20 chars
        w64le(minimal,20);minimal.insert(minimal.end(),arch,arch+20);
        w32le(minimal,8); // STRING
        const char* value="llama";
        w64le(minimal,5);minimal.insert(minimal.end(),value,value+5);
        while(minimal.size()%32)minimal.push_back(0);
        auto mem_ok=gguf::inspect_bytes(minimal.data(),minimal.size());
        CHECK(mem_ok.ok);
        if(mem_ok.ok){
            CHECK(mem_ok.config.architecture==model::Architecture::llama);
            CHECK(mem_ok.tensor_data_offset==minimal.size());
        }
    }

    // ROCmFP4 exact blocks (Codebook10 + UE4M3 scales, ciru-ai/ROCmFPX).
    // Scale byte 0x40: E=8, M=0 -> (8+0)*2^(8-10) = 2.0 (half-scale).
    // Code 7 = 10 (Codebook10 top level), code 9 = -1, code 0 = 0.
    {
        std::array<std::uint8_t,17> rf{};
        rf[16]=0x40;                       // UE4M3 half-scale = 2.0
        rf[0]=0x97;                        // lo=7 (+10) -> out[0]; hi=9 (-1) -> out[16]
        std::array<float,32> out{};
        CHECK(quant::dequantize(*quant::find("Q4_0_ROCMFP4_FAST"),rf.data(),32,out.data()));
        CHECK(out[0]==10.0f*2.0f);
        CHECK(out[16]==-1.0f*2.0f);
        for(std::size_t i=1;i<16;++i)CHECK(out[i]==0.0f);
        for(std::size_t i=17;i<32;++i)CHECK(out[i]==0.0f);
        CHECK(quant::row_bytes(*quant::find("Q4_0_ROCMFP4_FAST"),32)==17);
        CHECK(quant::row_bytes(*quant::find("Q4_0_ROCMFP4"),32)==18);
    }
    // Dual-scale: two halves with different scales; byte 0x7F scale decodes 0.
    {
        std::array<std::uint8_t,18> rd{};
        rd[0]=0x87;                        // lo nibble 7 (+10) -> out[0]; hi nibble 8 (-0) -> out[16]
        rd[16]=0x40;                       // half 0 scale 2.0
        rd[17]=0x7f;                       // half 1 scale = 0 (finite unsigned E4M3 NaN slot)
        std::array<float,32> out{};
        CHECK(quant::dequantize(*quant::find("Q4_0_ROCMFP4"),rd.data(),32,out.data()));
        CHECK(out[0]==10.0f*2.0f);
        CHECK(out[16]==0.0f);
        for(std::size_t i=1;i<16;++i)CHECK(out[i]==0.0f);
        for(std::size_t i=17;i<32;++i)CHECK(out[i]==0.0f);
    }

    if(failures){std::cerr<<failures<<" test(s) failed\n";return 1;}
    std::cout<<"ZenInfer core tests: PASS\n";return 0;
}
