// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/openvino.hpp"

namespace {
std::filesystem::path models_dir = "/opt/home/suvorova/projects/openvino/openvino.genai/.vscode/models";

ov::Core get_core() {
    static ov::Core core;
    return core;
}

void load_test(const std::string& model_path,
               bool use_cache = false,
               bool compile_only = false,
               bool import_only = false) {
    ov::AnyMap properties;
    if (use_cache) {
        properties[ov::cache_dir.name()] = std::string(models_dir / model_path / ".ov_genai_cache");
    }

    auto compile_model = [&]() {
        auto start = std::chrono::steady_clock::now();
        get_core().compile_model(models_dir / model_path / "openvino_model.xml", "CPU", properties);
        auto end = std::chrono::steady_clock::now();
        auto time_took = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Loading model: " << model_path << " took : " << time_took << std::endl;
    };

    auto read_and_compile_model = [&]() {
        // ov::AnyMap properties{
        //     {"NPU_USE_NPUW", "YES"},
        //     {"NPUW_DEVICES", "CPU"},
        //     {"NPUW_ONLINE_PIPELINE", "NONE"},
        //     {"BLOB_PATH", model_blob_path.string()},
        //     {"EXPORT_BLOB", true},
        //     {"CACHE_MODE", "OPTIMIZE_SPEED"},
        // };

        auto start = std::chrono::steady_clock::now();
        auto model = get_core().read_model(models_dir / model_path / "openvino_model.xml");
        get_core().compile_model(model, "CPU", properties);
        auto end = std::chrono::steady_clock::now();
        auto time_took = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Loading model: " << model_path << " took : " << time_took << std::endl;
    };

    auto import_model = [&]() {
        std::filesystem::path model_blob_path = models_dir / model_path / "openvino_model.blob";
        if (!std::filesystem::exists(model_blob_path)) {
            auto start_time = std::chrono::steady_clock::now();
            auto model = get_core().read_model(models_dir / model_path / "openvino_model.xml", {}, properties);
            auto compiled_model = get_core().compile_model(model, "CPU", properties);
            auto end_read_and_compile_time = std::chrono::steady_clock::now();
            std::cout
                << "Read and compile model took: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_read_and_compile_time - start_time).count()
                << " ms" << std::endl;

            auto start_export_time = std::chrono::steady_clock::now();
            std::ofstream model_blob(models_dir / model_path / "openvino_model.blob",
                                     std::ios::binary | std::ios_base::out);
            compiled_model.export_model(model_blob);
            model_blob.close();
            auto end_export_time = std::chrono::steady_clock::now();
            std::cout << "Export model took: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(end_export_time - start_time).count()
                      << " ms" << std::endl;
        }

        auto start = std::chrono::steady_clock::now();
        std::ifstream model_blob(models_dir / model_path / "openvino_model.blob", std::ios::binary | std::ios_base::in);
        get_core().import_model(model_blob, "CPU", properties);
        model_blob.close();
        auto end = std::chrono::steady_clock::now();
        auto time_took = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "Loading model: " << model_path << " took : " << time_took << std::endl;
    };

    if (import_only) {
        import_model();
    } else if (compile_only) {
        compile_model();
    } else {
        read_and_compile_model();
    }
}
}  // namespace

int main(int argc, char* argv[]) try {
    std::string model_a = "Qwen/Qwen2.5-7B-Instruct-int4";
    std::string model_b = "microsoft/Phi-3-mini-4k-instruct-int4";

    std::cout << "Use cache: false, compile only: false" << std::endl;
    for (size_t i = 0; i < 1; i++) {
        load_test(model_b);
        load_test(model_a);
    }
    std::cout << "Use cache: true, compile only: false" << std::endl;
    for (size_t i = 0; i < 4; i++) {
        load_test(model_b, true);
        load_test(model_a, true);
    }
    std::cout << "Use cache: false, compile only: true" << std::endl;
    for (size_t i = 0; i < 4; i++) {
        load_test(model_b, false, true);
        load_test(model_a, false, true);
    }
    std::cout << "Use cache: true, compile only: true" << std::endl;
    for (size_t i = 0; i < 4; i++) {
        load_test(model_b, true, true);
        load_test(model_a, true, true);
    }
    std::cout << "Use cache: false, compile only: false, import only: true" << std::endl;
    for (size_t i = 0; i < 4; i++) {
        load_test(model_b, false, false, true);
        load_test(model_a, false, false, true);
    }

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}

// LOAD TESTS
// Use cache: false, compile only: false
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 898
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 821
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 845
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 802
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 847
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 798
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 848
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 821
// Use cache: true, compile only: false
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 617
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 854
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 644
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 819
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 620
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 838
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 622
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 920
// Use cache: false, compile only: true
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 843
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 816
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 849
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 812
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 846
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 807
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 849
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 815
// Use cache: true, compile only: true
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 373
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 323
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 403
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 349
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 389
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 380
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 395
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 347

// PA, NO_CACHE
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 3009
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1774
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 966
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1761
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 945
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1763
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 964
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1775

// PA, CACHE
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 2985
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1790
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 941
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1751
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 943
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1762
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 942
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1815

// NO CACHE, extended logging
// Tokenizer ctor took 329 ms
// read model took 47 ms
// apply_paged_attention_transformations took 21 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 759 ms
// initialize_pipeline took 760 ms
// impl ctor took 783 ms
// ContinuousBatchingAdapter ctor took 836 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 1166 ms
// LLMPipeline ctor took 1166 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 1166
// Tokenizer ctor took 976 ms
// read model took 40 ms
// apply_paged_attention_transformations took 19 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 712 ms
// initialize_pipeline took 712 ms
// impl ctor took 733 ms
// ContinuousBatchingAdapter ctor took 779 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 1756 ms
// LLMPipeline ctor took 1756 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1756
// Tokenizer ctor took 101 ms
// read model took 92 ms
// apply_paged_attention_transformations took 23 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 760 ms
// initialize_pipeline took 760 ms
// impl ctor took 785 ms
// ContinuousBatchingAdapter ctor took 883 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 985 ms
// LLMPipeline ctor took 985 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 985
// Tokenizer ctor took 975 ms
// read model took 40 ms
// apply_paged_attention_transformations took 19 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 721 ms
// initialize_pipeline took 722 ms
// impl ctor took 742 ms
// ContinuousBatchingAdapter ctor took 789 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 1765 ms
// LLMPipeline ctor took 1765 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1765
// Tokenizer ctor took 100 ms
// read model took 49 ms
// apply_paged_attention_transformations took 21 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 762 ms
// initialize_pipeline took 763 ms
// impl ctor took 785 ms
// ContinuousBatchingAdapter ctor took 841 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 941 ms
// LLMPipeline ctor took 941 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 941
// Tokenizer ctor took 978 ms
// read model took 41 ms
// apply_paged_attention_transformations took 19 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 711 ms
// initialize_pipeline took 712 ms
// impl ctor took 733 ms
// ContinuousBatchingAdapter ctor took 780 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 1758 ms
// LLMPipeline ctor took 1758 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1758
// Tokenizer ctor took 100 ms
// read model took 46 ms
// apply_paged_attention_transformations took 21 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 761 ms
// initialize_pipeline took 762 ms
// impl ctor took 784 ms
// ContinuousBatchingAdapter ctor took 837 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 938 ms
// LLMPipeline ctor took 938 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 938
// Tokenizer ctor took 998 ms
// read model took 41 ms
// apply_paged_attention_transformations took 19 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 725 ms
// initialize_pipeline took 725 ms
// impl ctor took 746 ms
// ContinuousBatchingAdapter ctor took 794 ms
// ContinuousBatchingAdapter from LLMPipeline ctor took 1792 ms
// LLMPipeline ctor took 1792 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1792

// CACHE, extended logging
// read model took 47 ms
// apply_paged_attention_transformations took 21 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 532 ms
// initialize_pipeline took 533 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 2797
// read model took 44 ms
// apply_paged_attention_transformations took 21 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 689 ms
// initialize_pipeline took 690 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1849
// read model took 46 ms
// apply_paged_attention_transformations took 22 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 514 ms
// initialize_pipeline took 515 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 714
// read model took 41 ms
// apply_paged_attention_transformations took 19 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 692 ms
// initialize_pipeline took 692 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1835
// read model took 48 ms
// apply_paged_attention_transformations took 22 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 534 ms
// initialize_pipeline took 534 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 736
// read model took 41 ms
// apply_paged_attention_transformations took 19 ms
// apply_gather_before_matmul_transformation took 0 ms
// compile_model took 657 ms
// initialize_pipeline took 658 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1788
// read model took 46 ms
// apply_paged_attention_transformations took 21 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 538 ms
// initialize_pipeline took 539 ms
// Loading model: microsoft/Phi-3-mini-4k-instruct-int4 took : 739
// read model took 45 ms
// apply_paged_attention_transformations took 21 ms
// apply_gather_before_matmul_transformation took 1 ms
// compile_model took 705 ms
// initialize_pipeline took 706 ms
// Loading model: Qwen/Qwen2.5-7B-Instruct-int4 took : 1858