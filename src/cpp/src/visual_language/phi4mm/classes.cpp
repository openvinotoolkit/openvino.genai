// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/phi4mm/classes.hpp"
#include <algorithm>

#include "visual_language/clip.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "openvino/opsets/opset13.hpp"
#include "utils.hpp"

namespace {

const std::regex NATIVE_PATTERN{R"(<\|image_(\d+)\|>)"};

// Similar to Phi3, but without newline after image tag
void write_native(std::ostream& os, size_t idx) {
    os << "<|image_" << idx + 1 << "|>";
}

std::unique_ptr<ov::genai::CircularBufferQueue<ov::InferRequest>> create_image_preprocessors() {
    using namespace ov;
    using namespace element;
    using namespace opset13;
    using namespace std;

    auto t0 = make_shared<Parameter>(u8, PartialShape{-1, -1, -1, -1});  //  -> u8[?,?,?,?]
    t0->output(0).get_tensor().set_names({"image"});
    auto t1 = make_shared<Parameter>(f32, PartialShape{-1, -1});  //  -> f32[?,?]
    t1->output(0).get_tensor().set_names({"attention_mask"});
    auto t2 = make_shared<Parameter>(i64, PartialShape{-1});    //  -> i64[?]
    t2->output(0).get_tensor().set_names({"new_size"});
    auto t3 = make_shared<Parameter>(i64, PartialShape{});      //  -> i64[]
    t3->output(0).get_tensor().set_names({"7", "padding_width"});
    auto t4 = make_shared<Parameter>(i64, PartialShape{});      //  -> i64[]
    t4->output(0).get_tensor().set_names({"6", "padding_height"});
    auto t5 = make_shared<Constant>(i64, Shape{1}, 1);          //  -> i64[1]([1])
    auto t6 = make_shared<ShapeOf>(t1);                         // f32[?,?] -> i64[2]
    auto t7 = make_shared<Constant>(i64, Shape{}, 0);           //  -> i64[](0)
    auto t8 = make_shared<Constant>(i64, Shape{}, 0);           //  -> i64[](0)
    auto t9 = make_shared<Gather>(t6, t7, t8);                  // i64[2], i64[], i64[] -> i64[]
    auto t10 = make_shared<Constant>(i64, Shape{}, 32);         //  -> i64[](32)
    auto t11 = make_shared<Divide>(t9, t10, "numpy");           // i64[], i64[] -> i64[]
    auto t12 = make_shared<Floor>(t11);                         // i64[] -> i64[]
    auto t13 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t14 = make_shared<Unsqueeze>(t12, t13);                // i64[], i32[] -> i64[1]
    auto t15 = make_shared<Constant>(i64, Shape{1}, 32);        //  -> i64[1]([32])
    auto t16 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t17 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t18 = make_shared<Gather>(t6, t16, t17);               // i64[2], i64[], i32[] -> i64[]
    auto t19 = make_shared<Divide>(t18, t10, "numpy");          // i64[], i64[] -> i64[]
    auto t20 = make_shared<Floor>(t19);                         // i64[] -> i64[]
    auto t21 = make_shared<Unsqueeze>(t20, t13);                // i64[], i32[] -> i64[1]
    auto t22 = make_shared<Constant>(i64, Shape{1}, 32);        //  -> i64[1]([32])
    auto t23 = make_shared<Concat>(NodeVector{t5, t14, t15, t21, t22}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t24 = make_shared<Reshape>(t1, t23, false);            // f32[?,?], i64[5] -> f32[?,?,?,?,?]
    auto t25 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t26 = make_shared<Transpose>(t24, t25);                // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t27 = make_shared<Constant>(i64, Shape{3}, vector<int64_t>{-1, 32, 32});  //  -> i64[3]([-1, 32, 32])
    auto t28 = make_shared<Reshape>(t26, t27, false);           // f32[?,?,?,?,?], i64[3] -> f32[?,32,32]
    auto t29 = make_shared<Constant>(i64, Shape{2}, 0);         //  -> i64[2]([0, 0])
    auto t30 = make_shared<Constant>(i64, Shape{2}, 9223372036854775807);  //  -> i64[2]([9223372036854775807, 9223372036854775807])
    auto t31 = make_shared<Constant>(i64, Shape{2}, 2);         //  -> i64[2]([2, 2])
    auto t32 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 2});  //  -> i64[2]([1, 2])
    auto t33 = make_shared<Slice>(t28, t29, t30, t31, t32);     // f32[?,32,32], i64[2], i64[2], i64[2], i64[2] -> f32[?,16,16]
    auto t34 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t35 = make_shared<Constant>(i64, Shape{1}, 16);        //  -> i64[1]([16])
    auto t36 = make_shared<Constant>(i64, Shape{1}, 16);        //  -> i64[1]([16])
    auto t37 = make_shared<Concat>(NodeVector{t34, t14, t21, t35, t36}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t38 = make_shared<Reshape>(t33, t37, false);           // f32[?,16,16], i64[5] -> f32[?,?,?,?,?]
    auto t39 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t40 = make_shared<Transpose>(t38, t39);                // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t41 = make_shared<ShapeOf>(t40);                       // f32[?,?,?,?,?] -> i64[5]
    auto t42 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t43 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t44 = make_shared<Gather>(t41, t42, t43);              // i64[5], i64[1], i64[] -> i64[1]
    auto t45 = make_shared<Constant>(i64, Shape{1}, 2);         //  -> i64[1]([2])
    auto t46 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t47 = make_shared<Gather>(t41, t45, t46);              // i64[5], i64[1], i64[] -> i64[1]
    auto t48 = make_shared<Multiply>(t44, t47, "numpy");        // i64[1], i64[1] -> i64[1]
    auto t49 = make_shared<Constant>(i64, Shape{1}, 3);         //  -> i64[1]([3])
    auto t50 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t51 = make_shared<Gather>(t41, t49, t50);              // i64[5], i64[1], i64[] -> i64[1]
    auto t52 = make_shared<Constant>(i64, Shape{1}, 4);         //  -> i64[1]([4])
    auto t53 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t54 = make_shared<Gather>(t41, t52, t53);              // i64[5], i64[1], i64[] -> i64[1]
    auto t55 = make_shared<Multiply>(t51, t54, "numpy");        // i64[1], i64[1] -> i64[1]
    auto t56 = make_shared<Concat>(NodeVector{t48, t55}, 0);    // i64[1], i64[1] -> i64[2]
    auto t57 = make_shared<Reshape>(t40, t56, false);           // f32[?,?,?,?,?], i64[2] -> f32[?,?]
    auto t58 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{0, 1});  //  -> i32[2]([0, 1])
    auto t59 = make_shared<ReduceSum>(t57, t58);                // f32[?,?], i32[2] -> f32[]
    auto t60 = make_shared<Constant>(f32, Shape{}, 257.0);      //  -> f32[](257.0)
    auto t61 = make_shared<Add>(t59, t60);                      // f32[], f32[] -> f32[]
    auto t62 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t63 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t64 = make_shared<Gather>(t57, t62, t63);              // f32[?,?], i64[], i64[] -> f32[?]
    auto t65 = make_shared<Constant>(i32, Shape{1}, 0);         //  -> i32[1]([0])
    auto t66 = make_shared<ReduceSum>(t64, t65);                // f32[?], i32[1] -> f32[]
    auto t67 = make_shared<Add>(t61, t66);                      // f32[], f32[] -> f32[]
    auto t68 = make_shared<Constant>(f32, Shape{}, 16.0);       //  -> f32[](16.0)
    auto t69 = make_shared<Add>(t67, t68);                      // f32[], f32[] -> f32[]
    auto t70 = make_shared<Convert>(t69, i64);                  // f32[] -> i64[]
    auto t71 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t72 = make_shared<Unsqueeze>(t70, t71);                // i64[], i64[] -> i64[1]
    auto t73 = make_shared<Constant>(f32, Shape{1, 32, 32}, 1.0f);  //  -> f32[1,32,32]
    auto t74 = make_shared<Concat>(NodeVector{t73, t28}, 0);    // f32[1,32,32], f32[?,32,32] -> f32[1..,32,32]
    auto t75 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t76 = make_shared<Unsqueeze>(t74, t75);                // f32[1..,32,32], i64[] -> f32[1,1..,32,32]
    auto t77 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t78 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t79 = make_shared<Gather>(t0, t77, t78);               // u8[?,?,?,?], i64[], i64[] -> u8[?,?,?]
    auto t80 = make_shared<Convert>(t79, f32);                  // u8[?,?,?] -> f32[?,?,?]
    auto t81 = make_shared<Constant>(f32, Shape{1, 1, 1}, 255.0f);  //  -> f32[1,1,1]([[[255.0]]])
    auto t82 = make_shared<Divide>(t80, t81, "numpy");          // f32[?,?,?], f32[1,1,1] -> f32[?,?,?]
    auto t83 = make_shared<Constant>(i64, Shape{3}, vector<int64_t>{2, 0, 1});  //  -> i64[3]([2, 0, 1])
    auto t84 = make_shared<Transpose>(t82, t83);                // f32[?,?,?], i64[3] -> f32[?,?,?]
    auto t85 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t86 = make_shared<ShapeOf>(t84);                       // f32[?,?,?] -> i64[3]
    auto t87 = make_shared<Concat>(NodeVector{t85, t86}, 0);    // i64[1], i64[3] -> i64[4]
    auto t88 = make_shared<Reshape>(t84, t87, false);           // f32[?,?,?], i64[4] -> f32[?,?,?,?]
    auto t89 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 0});  //  -> i64[2]([1, 0])
    auto t90 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t91 = make_shared<Gather>(t2, t89, t90);               // i64[?], i64[2], i64[] -> i64[2]
    auto t92 = make_shared<Convert>(t91, i32);                  // i64[2] -> i32[2]
    auto t93 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t94_attrs{Interpolate::InterpolateMode::BILINEAR_PILLOW, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t94 = make_shared<Interpolate>(t88, t92, t93, t94_attrs);  // f32[?,?,?,?], i32[2], i32[2] -> f32[?,?,?,?]
    auto t95 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t96 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t97 = make_shared<Gather>(t86, t95, t96);              // i64[3], i64[1], i32[] -> i64[1]
    auto t98 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 0});  //  -> i64[2]([1, 0])
    auto t99 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t100 = make_shared<Gather>(t2, t98, t99);              // i64[?], i64[2], i64[] -> i64[2]
    auto t101 = make_shared<Concat>(NodeVector{t97, t100}, 0);  // i64[1], i64[2] -> i64[3]
    auto t102 = make_shared<Reshape>(t94, t101, false);         // f32[?,?,?,?], i64[3] -> f32[?,?,?]
    auto t103 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t104 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t105 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t106 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t107 = make_shared<Gather>(t2, t105, t106);            // i64[?], i64[1], i64[] -> i64[1]
    auto t108 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t109 = make_shared<Unsqueeze>(t3, t108);               // i64[], i32[] -> i64[1]
    auto t110 = make_shared<Concat>(NodeVector{t104, t107, t109}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t111 = make_shared<Broadcast>(t103, t110);             // f32[], i64[3] -> f32[3,?,?]
    auto t112 = make_shared<Concat>(NodeVector{t102, t111}, 2);  // f32[?,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t113 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t114 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t115 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t116 = make_shared<Unsqueeze>(t4, t115);               // i64[], i32[] -> i64[1]
    auto t117 = make_shared<Constant>(i64, Shape{1}, 0);        //  -> i64[1]([0])
    auto t118 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t119 = make_shared<Gather>(t2, t117, t118);            // i64[?], i64[1], i64[] -> i64[1]
    auto t120 = make_shared<Add>(t109, t119);                   // i64[1], i64[1] -> i64[1]
    auto t121 = make_shared<Concat>(NodeVector{t114, t116, t120}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t122 = make_shared<Broadcast>(t113, t121);             // f32[], i64[3] -> f32[3,?,?]
    auto t123 = make_shared<Concat>(NodeVector{t112, t122}, 1);  // f32[3,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t124 = make_shared<Constant>(f32, Shape{3, 1, 1}, -0.5f);  //  -> f32[3,1,1]([[[-0.5]], [[-0.5]], [[-0.5]]])
    auto t125 = make_shared<Add>(t123, t124);                   // f32[3,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t126 = make_shared<Constant>(f32, Shape{3, 1, 1}, 0.5f);  //  -> f32[3,1,1]([[[0.5]], [[0.5]], [[0.5]]])
    auto t127 = make_shared<Divide>(t125, t126, "numpy");       // f32[3,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t128 = make_shared<ShapeOf>(t127);                     // f32[3,?,?] -> i64[3]
    auto t129 = make_shared<Constant>(i64, Shape{}, 2);         //  -> i64[](2)
    auto t130 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t131 = make_shared<Gather>(t128, t129, t130);          // i64[3], i64[], i32[] -> i64[]
    auto t132 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t133 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t134 = make_shared<Gather>(t128, t132, t133);          // i64[3], i64[], i64[] -> i64[]
    auto t135 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t136 = make_shared<Unsqueeze>(t127, t135);             // f32[3,?,?], i64[] -> f32[1,3,?,?]
    auto t137 = make_shared<Constant>(i32, Shape{2}, 448);      //  -> i32[2]([448, 448])
    auto t138 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t139_attrs{Interpolate::InterpolateMode::CUBIC, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t139 = make_shared<Interpolate>(t136, t137, t138, t139_attrs);  // f32[1,3,?,?], i32[2], i32[2] -> f32[1,3,448,448]
    auto t140 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t141 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t142 = make_shared<Constant>(i64, Shape{}, 448);       //  -> i64[](448)
    auto t143 = make_shared<Divide>(t134, t142, "numpy");       // i64[], i64[] -> i64[]
    auto t144 = make_shared<Floor>(t143);                       // i64[] -> i64[]
    auto t145 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t146 = make_shared<Unsqueeze>(t144, t145);             // i64[], i32[] -> i64[1]
    auto t147 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t148 = make_shared<Divide>(t131, t142, "numpy");       // i64[], i64[] -> i64[]
    auto t149 = make_shared<Floor>(t148);                       // i64[] -> i64[]
    auto t150 = make_shared<Unsqueeze>(t149, t145);             // i64[], i32[] -> i64[1]
    auto t151 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t152 = make_shared<Concat>(NodeVector{t140, t141, t146, t147, t150, t151}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[6]
    auto t153 = make_shared<Reshape>(t127, t152, false);        // f32[3,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t154 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 2, 4, 1, 3, 5});  //  -> i64[6]([0, 2, 4, 1, 3, 5])
    auto t155 = make_shared<Transpose>(t153, t154);             // f32[?,?,?,?,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t156 = make_shared<Constant>(i64, Shape{4}, vector<int64_t>{-1, 3, 448, 448});  //  -> i64[4]([-1, 3, 448, 448])
    auto t157 = make_shared<Reshape>(t155, t156, false);        // f32[?,?,?,?,?,?], i64[4] -> f32[?,3,448,448]
    auto t158 = make_shared<Concat>(NodeVector{t139, t157}, 0);  // f32[1,3,448,448], f32[?,3,448,448] -> f32[1..,3,448,448]
    auto t159 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t160 = make_shared<Unsqueeze>(t158, t159);             // f32[1..,3,448,448], i64[] -> f32[1,1..,3,448,448]
    auto t161 = make_shared<Result>(t72);                       // i64[1] -> i64[1]
    t161->output(0).get_tensor().set_names({"num_img_tokens"});
    auto t162 = make_shared<Result>(t76);                       // f32[1,1..,32,32] -> f32[1,1..,32,32]
    t162->output(0).get_tensor().set_names({"image_attention_mask"});
    auto t163 = make_shared<Result>(t131);                      // i64[] -> i64[]
    t163->output(0).get_tensor().set_names({"155", "image_width"});
    auto t164 = make_shared<Result>(t134);                      // i64[] -> i64[]
    t164->output(0).get_tensor().set_names({"152", "image_height"});
    auto t165 = make_shared<Result>(t160);                      // f32[1,1..,3,448,448] -> f32[1,1..,3,448,448]
    t165->output(0).get_tensor().set_names({"input_image_embeds"});

    ResultVector results{t165, t164, t163, t162, t161};
    SinkVector sinks{};
    ParameterVector parameters{t0, t1, t2, t3, t4};
    auto model = make_shared<Model>(results, sinks, parameters);
    using namespace ov::genai;
    CompiledModel compiled = utils::singleton_core().compile_model(model, "CPU");
    return make_unique<CircularBufferQueue<InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        }
    );
}

std::unique_ptr<ov::genai::CircularBufferQueue<ov::InferRequest>> create_patch_position_ids_model() {
    using namespace ov;
    using namespace element;
    using namespace opset13;
    using namespace std;

    auto t0 = make_shared<Parameter>(f32, PartialShape{-1, -1, -1, -1, -1});  //  -> f32[?,?,?,?,?]
    t0->output(0).get_tensor().set_names({"input_image_embeds"});
    auto t1 = make_shared<Parameter>(f32, PartialShape{-1, -1, -1, -1});  //  -> f32[?,?,?,?]
    t1->output(0).get_tensor().set_names({"image_attention_mask"});
    auto t2 = make_shared<Constant>(i64, Shape{1}, -1);         //  -> i32[1]([-1])
    auto t3 = make_shared<ShapeOf>(t1);                         // f32[?,?,?,?] -> i32[4]
    auto t4 = make_shared<Constant>(i32, Shape{1}, 2);          //  -> i32[1]([2])
    auto t5 = make_shared<Constant>(i32, Shape{1}, 2147483647);  //  -> i32[1]([2147483647])
    auto t6 = make_shared<Constant>(i32, Shape{1}, 1);          //  -> i32[1]([1])
    auto t7 = make_shared<Constant>(i32, Shape{1}, 0);          //  -> i64[1]([0])
    auto t8 = make_shared<Slice>(t3, t4, t5, t6, t7);           // i32[4], i32[1], i32[1], i32[1], i64[1] -> i32[2]
    auto t9 = make_shared<Concat>(NodeVector{t2, t8}, 0);       // i32[1], i32[2] -> i32[3]
    auto t10 = make_shared<Reshape>(t1, t9, true);              // f32[?,?,?,?], i32[3] -> f32[?,?,?]
    auto t11 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{0, -1});  //  -> i64[2]([0, -1])
    auto t12 = make_shared<Reshape>(t10, t11, true);            // f32[?,?,?], i64[2] -> f32[?,?]
    auto t13 = make_shared<Constant>(f32, Shape{32}, vector<float>{0.0, 0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.53125, 0.5625, 0.59375, 0.625, 0.65625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 0.96875});  //  -> f32[32]
    auto t14 = make_shared<Constant>(f32, Shape{31}, vector<float>{0.03125, 0.0625, 0.09375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.53125, 0.5625, 0.59375, 0.625, 0.65625, 0.6875, 0.71875, 0.75, 0.78125, 0.8125, 0.84375, 0.875, 0.90625, 0.9375, 0.96875});  //  -> f32[31]
    auto t15 = make_shared<Bucketize>(t13, t14, i64, false);    // f32[32], f32[31] -> i64[32]
    auto t16 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t17 = make_shared<Unsqueeze>(t15, t16);                // i64[32], i64[] -> i64[32,1]
    auto t18 = make_shared<Constant>(i64, Shape{1, 1}, 32);     //  -> i64[1,1]([[32]])
    auto t19 = make_shared<Multiply>(t17, t18, "numpy");        // i64[32,1], i64[1,1] -> i64[32,1]
    auto t20 = make_shared<Add>(t19, t15);                      // i64[32,1], i64[32] -> i64[32,32]
    auto t21 = make_shared<Constant>(i32, Shape{1}, -1);        //  -> i32[1]([-1])
    auto t22 = make_shared<Reshape>(t20, t21, true);            // i64[32,32], i32[1] -> i64[1024]
    auto t23 = make_shared<ShapeOf>(t10);                       // f32[?,?,?] -> i64[3]
    auto t24 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t25 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t26 = make_shared<Gather>(t23, t24, t25);              // i64[3], i64[1], i64[] -> i64[1]
    auto t27 = make_shared<Tile>(t22, t26);                     // i64[1024], i64[1] -> i64[?]
    auto t28 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t29 = make_shared<Concat>(NodeVector{t26, t28}, 0);    // i64[1], i64[1] -> i64[2]
    auto t30 = make_shared<Reshape>(t27, t29, false);           // i64[?], i64[2] -> i64[?,?]
    auto t31 = make_shared<Convert>(t30, f32);                  // i64[?,?] -> f32[?,?]
    auto t32 = make_shared<Multiply>(t12, t31, "numpy");        // f32[?,?], f32[?,?] -> f32[?,?]
    auto t33 = make_shared<Result>(t32);                        // f32[?,?] -> f32[?,?]
    t33->output(0).get_tensor().set_names({"patch_position_ids"});

    ResultVector results{t33};
    SinkVector sinks{};
    ParameterVector parameters{t0, t1};
    auto model = make_shared<Model>(results, sinks, parameters);
    using namespace ov::genai;
    CompiledModel compiled = utils::singleton_core().compile_model(model, "CPU");
    return make_unique<CircularBufferQueue<InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        }
    );
}

std::unique_ptr<ov::genai::CircularBufferQueue<ov::InferRequest>> create_separator_inserters() {
    using namespace ov;
    using namespace element;
    using namespace opset13;
    using namespace std;

    auto t0 = make_shared<Parameter>(f32, PartialShape{1, -1, -1, -1});  //  -> f32[1,?,?,?]
    t0->output(0).get_tensor().set_names({"img_features"});
    auto t1 = make_shared<Parameter>(i32, PartialShape{});      //  -> i32[]
    t1->output(0).get_tensor().set_names({"height"});
    auto t2 = make_shared<Parameter>(i32, PartialShape{});      //  -> i32[]
    t2->output(0).get_tensor().set_names({"width"});
    auto t3 = make_shared<Parameter>(f32, PartialShape{1, 1, 1, -1});  //  -> f32[1,1,1,?]
    t3->output(0).get_tensor().set_names({"sub_GN"});
    auto t4 = make_shared<Parameter>(f32, PartialShape{1, 1, -1});  //  -> f32[1,1,?]
    t4->output(0).get_tensor().set_names({"glb_GN"});
    auto t5 = make_shared<Constant>(i64, Shape{}, 0);           //  -> i64[](0)
    auto t6 = make_shared<Constant>(i64, Shape{}, 0);           //  -> i64[](0)
    auto t7 = make_shared<Gather>(t0, t5, t6);                  // f32[1,?,?,?], i64[], i64[] -> f32[?,?,?]
    auto t8 = make_shared<Constant>(i64, Shape{1}, 1);          //  -> i64[1]([1])
    auto t9 = make_shared<Constant>(i64, Shape{1}, -1);         //  -> i64[1]([-1])
    auto t10 = make_shared<Constant>(i64, Shape{1}, 256);       //  -> i64[1]([256])
    auto t11 = make_shared<ShapeOf>(t0);                        // f32[1,?,?,?] -> i64[4]
    auto t12 = make_shared<Constant>(i64, Shape{1}, 3);         //  -> i64[1]([3])
    auto t13 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t14 = make_shared<Gather>(t11, t12, t13);              // i64[4], i64[1], i64[] -> i64[1]
    auto t15 = make_shared<Concat>(NodeVector{t8, t9, t10, t14}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t16 = make_shared<Reshape>(t7, t15, false);            // f32[?,?,?], i64[4] -> f32[1,?,256,?]
    auto t17 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t18 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t19 = make_shared<Gather>(t16, t17, t18);              // f32[1,?,256,?], i64[], i64[] -> f32[?,256,?]
    auto t20 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t21 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t22 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t23 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t24 = make_shared<Slice>(t19, t20, t21, t22, t23);     // f32[?,256,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,256,?]
    auto t25 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t26 = make_shared<Convert>(t1, i64);                   // i32[] -> i64[]
    auto t27 = make_shared<Constant>(i64, Shape{}, 448);        //  -> i64[](448)
    auto t28 = make_shared<Divide>(t26, t27, "numpy");          // i64[], i64[] -> i64[]
    auto t29 = make_shared<Floor>(t28);                         // i64[] -> i64[]
    auto t30 = make_shared<Convert>(t2, i64);                   // i32[] -> i64[]
    auto t31 = make_shared<Constant>(i64, Shape{}, 448);        //  -> i64[](448)
    auto t32 = make_shared<Divide>(t30, t31, "numpy");          // i64[], i64[] -> i64[]
    auto t33 = make_shared<Floor>(t32);                         // i64[] -> i64[]
    auto t34 = make_shared<Multiply>(t29, t33, "numpy");        // i64[], i64[] -> i64[]
    auto t35 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t36 = make_shared<Reshape>(t34, t35, false);           // i64[], i64[1] -> i64[1]
    auto t37 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t38 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t39 = make_shared<Slice>(t24, t25, t36, t37, t38);     // f32[?,256,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,256,?]
    auto t40 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t41 = make_shared<Unsqueeze>(t29, t40);                // i64[], i32[] -> i64[1]
    auto t42 = make_shared<Unsqueeze>(t33, t40);                // i64[], i32[] -> i64[1]
    auto t43 = make_shared<Constant>(i64, Shape{1}, 16);        //  -> i64[1]([16])
    auto t44 = make_shared<Constant>(i64, Shape{1}, 16);        //  -> i64[1]([16])
    auto t45 = make_shared<Concat>(NodeVector{t41, t42, t43, t44, t14}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t46 = make_shared<Reshape>(t39, t45, false);           // f32[?,256,?], i64[5] -> f32[?,?,?,?,?]
    auto t47 = make_shared<Constant>(i32, Shape{5}, vector<int32_t>{0, 2, 1, 3, 4});  //  -> i32[5]([0, 2, 1, 3, 4])
    auto t48 = make_shared<Transpose>(t46, t47);                // f32[?,?,?,?,?], i32[5] -> f32[?,?,?,?,?]
    auto t49 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t50 = make_shared<Constant>(i64, Shape{}, 16);         //  -> i64[](16)
    auto t51 = make_shared<Multiply>(t29, t50, "numpy");        // i64[], i64[] -> i64[]
    auto t52 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t53 = make_shared<Unsqueeze>(t51, t52);                // i64[], i32[] -> i64[1]
    auto t54 = make_shared<Constant>(i64, Shape{}, 16);         //  -> i64[](16)
    auto t55 = make_shared<Multiply>(t33, t54, "numpy");        // i64[], i64[] -> i64[]
    auto t56 = make_shared<Unsqueeze>(t55, t52);                // i64[], i32[] -> i64[1]
    auto t57 = make_shared<Concat>(NodeVector{t49, t53, t56, t14}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t58 = make_shared<Reshape>(t48, t57, false);           // f32[?,?,?,?,?], i64[4] -> f32[?,?,?,?]
    auto t59 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t60 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t61 = make_shared<Reshape>(t51, t60, false);           // i64[], i64[1] -> i64[1]
    auto t62 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t63 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t64 = make_shared<Concat>(NodeVector{t59, t61, t62, t63}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t65 = make_shared<Tile>(t3, t64);                      // f32[1,1,1,?], i64[4] -> f32[?,?,?,?]
    auto t66 = make_shared<Concat>(NodeVector{t58, t65}, 2);    // f32[?,?,?,?], f32[?,?,?,?] -> f32[?,?,?,?]
    auto t67 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t68 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t69 = make_shared<Concat>(NodeVector{t67, t68, t14}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t70 = make_shared<Reshape>(t66, t69, false);           // f32[?,?,?,?], i64[3] -> f32[1,?,?]
    auto t71 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t72 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t73 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t74 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t75 = make_shared<Slice>(t19, t71, t72, t73, t74);     // f32[?,256,?], i64[1], i64[1], i64[1], i64[1] -> f32[..1,256,?]
    auto t76 = make_shared<Constant>(i64, Shape{4}, vector<int64_t>{1, 16, 16, -1});  //  -> i64[4]([1, 16, 16, -1])
    auto t77 = make_shared<Reshape>(t75, t76, true);            // f32[..1,256,?], i64[4] -> f32[1,16,16,?]
    auto t78 = make_shared<Constant>(i64, Shape{4}, vector<int64_t>{1, 16, 1, 1});  //  -> i64[4]([1, 16, 1, 1])
    auto t79 = make_shared<Tile>(t3, t78);                      // f32[1,1,1,?], i64[4] -> f32[1,16,1,?]
    auto t80 = make_shared<Concat>(NodeVector{t77, t79}, 2);    // f32[1,16,16,?], f32[1,16,1,?] -> f32[1,16,17,?]
    auto t81 = make_shared<Reshape>(t80, t69, false);           // f32[1,16,17,?], i64[3] -> f32[1,?,?]
    auto t82 = make_shared<Concat>(NodeVector{t70, t4, t81}, 1);  // f32[1,?,?], f32[1,1,?], f32[1,?,?] -> f32[1,1..,?]
    auto t83 = make_shared<Result>(t82);                        // f32[1,1..,?] -> f32[1,1..,?]
    t83->output(0).get_tensor().set_names({});

    ResultVector results{t83};
    SinkVector sinks{};
    ParameterVector parameters{t0, t1, t2, t3, t4};
    auto model = make_shared<Model>(results, sinks, parameters);
    using namespace ov::genai;
    CompiledModel compiled = utils::singleton_core().compile_model(model, "CPU");
    return make_unique<CircularBufferQueue<InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        }
    );
}

struct TargetSizes {
    size_t width;
    size_t height;
    size_t padding_width;
    size_t padding_height;
    ov::Tensor attention_mask;
};

TargetSizes get_target_sizes(const ov::Tensor& image, const size_t dynamic_hd = 36) {
    constexpr size_t DYHD_BASE_RESOLUTION = 448;

    constexpr size_t MASK_RESOLUTION = DYHD_BASE_RESOLUTION / 14;
    // nhwc
    const auto image_shape = image.get_shape();

    const size_t orig_width = image_shape[2];
    const size_t orig_height = image_shape[1];

    const size_t w_crop_num = static_cast<size_t>(std::ceil(static_cast<float>(orig_width) / DYHD_BASE_RESOLUTION));
    const size_t h_crop_num = static_cast<size_t>(std::ceil(static_cast<float>(orig_height) / DYHD_BASE_RESOLUTION));

    std::pair<size_t, size_t> target_aspect_ratio{1, 1};
    size_t target_width;
    size_t target_height;
    if (w_crop_num * h_crop_num > dynamic_hd) {
        const size_t min_num = 1;
        const size_t max_num = dynamic_hd;
        float aspect_ratio = static_cast<float>(orig_width) / orig_height;

        std::set<std::pair<size_t, size_t>> target_ratios;
        for (size_t i = 1; i <= max_num; ++i) {
            for (size_t j = 1; j <= max_num; ++j) {
                if (i * j <= max_num && i * j >= min_num) {
                    target_ratios.emplace(i, j);
                }
            }
        }

        std::vector<std::pair<size_t, size_t>> target_ratios_vec(target_ratios.begin(), target_ratios.end());
        std::sort(target_ratios_vec.begin(),
                  target_ratios_vec.end(),
                  [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
                      return a.first * a.second < b.first * b.second;
                  });

        // Find the closest aspect ratio to the target
        float best_ratio_diff = std::numeric_limits<float>::infinity();
        for (const auto& ratio : target_ratios_vec) {
            const float target_aspect_ratio_value = static_cast<float>(ratio.first) / ratio.second;
            const float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio_value);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                target_aspect_ratio = ratio;
            } else if (ratio_diff == best_ratio_diff && orig_width * orig_height > 0.5f * DYHD_BASE_RESOLUTION * DYHD_BASE_RESOLUTION * ratio.first * ratio.second) {
                target_aspect_ratio = ratio;
            }
        }

        // Calculate the target width and height
        target_width = DYHD_BASE_RESOLUTION * target_aspect_ratio.first;
        target_height = DYHD_BASE_RESOLUTION * target_aspect_ratio.second;
    } else {
        target_width = DYHD_BASE_RESOLUTION * w_crop_num;
        target_height = DYHD_BASE_RESOLUTION * h_crop_num;
        target_aspect_ratio = {w_crop_num, h_crop_num};
    }

    // Calculate the ratio
    float ratio_width = static_cast<float>(target_width) / orig_width;
    float ratio_height = static_cast<float>(target_height) / orig_height;

    // width, height
    std::pair<size_t, size_t> new_size;
    size_t padding_width = 0;
    size_t padding_height = 0;
    if (ratio_width < ratio_height) {
        new_size = {target_width, static_cast<size_t>(orig_height * ratio_width)};
        padding_width = 0;
        padding_height = target_height - static_cast<size_t>(orig_height * ratio_width);
    } else {
        new_size = {static_cast<size_t>(orig_width * ratio_height), target_height};
        padding_width = target_width - static_cast<size_t>(orig_width * ratio_height);
        padding_height = 0;
    }

    // todo: implement as int32 or bool mask
    ov::Tensor attention_mask{
        ov::element::f32,
        {MASK_RESOLUTION * target_aspect_ratio.second, MASK_RESOLUTION * target_aspect_ratio.first}
    };
    std::fill(attention_mask.data<float>(), attention_mask.data<float>() + attention_mask.get_size(), 1.0f);

    if (padding_width >= 14) {
        size_t padding_width_blocks = static_cast<size_t>(static_cast<float>(padding_width) / 14);
        size_t row_width = MASK_RESOLUTION * target_aspect_ratio.first;
        size_t column_height = MASK_RESOLUTION * target_aspect_ratio.second;
        for (size_t i = 0; i < column_height; ++i) {
            std::fill_n(attention_mask.data<float>() + (i * row_width) + (row_width - padding_width_blocks), padding_width_blocks, 0.0f);
        }
    }
    if (padding_height >= 14) {
        size_t row_width = MASK_RESOLUTION * target_aspect_ratio.first;
        size_t column_height = MASK_RESOLUTION * target_aspect_ratio.second;
        size_t padding_height_blocks = static_cast<size_t>(static_cast<float>(padding_height) / 14);
        std::fill(attention_mask.data<float>() + (column_height - padding_height_blocks) * row_width, attention_mask.data<float>() + (row_width * column_height), 0.0f);
    }

    return {
        new_size.first, // width
        new_size.second, // height
        padding_width,
        padding_height,
        attention_mask,
    };
}

} // namespace

namespace ov::genai {

VisionEncoderPhi4MM::VisionEncoderPhi4MM(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties
) :
VisionEncoder(model_dir, device, properties),
m_image_preprocessors{create_image_preprocessors()},
m_patch_position_ids_model{create_patch_position_ids_model()},
m_separator_inserters{create_separator_inserters()} {
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_projection_model.xml", device, {});
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
}

VisionEncoderPhi4MM::VisionEncoderPhi4MM(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap properties
) : 
VisionEncoder(models_map, config_dir_path, device, properties),
m_image_preprocessors{create_image_preprocessors()},
m_patch_position_ids_model{create_patch_position_ids_model()},
m_separator_inserters{create_separator_inserters()} {
    const auto& vision_projection_model = utils::get_model_weights_pair(models_map, "vision_projection").first;
    const auto& vision_projection_weights = utils::get_model_weights_pair(models_map, "vision_projection").second;
    auto compiled_model = utils::singleton_core().compile_model(vision_projection_model, vision_projection_weights, device, properties);
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
}

EncodedImage VisionEncoderPhi4MM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);
    ov::Tensor input_image_embeds{ov::element::f32, {}}, image_attention_mask{ov::element::f32, {}}, patch_position_ids{ov::element::f32, {}};
    int32_t image_height = 0, image_width = 0, num_img_tokens = 0;

    auto target_sizes = get_target_sizes(image);
    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_image_preprocessors.get()};
        ov::InferRequest& image_preprocessor = lock.get();
        image_preprocessor.set_tensor("image", image);
        
        ov::Tensor new_size_tensor{ov::element::i64, {2}};
        new_size_tensor.data<int64_t>()[0] = target_sizes.width;
        new_size_tensor.data<int64_t>()[1] = target_sizes.height;
        image_preprocessor.set_tensor("new_size", new_size_tensor);

        ov::Tensor padding_width_tensor{ov::element::i64, {}};
        padding_width_tensor.data<int64_t>()[0] = target_sizes.padding_width;
        image_preprocessor.set_tensor("padding_width", padding_width_tensor);

        ov::Tensor padding_height_tensor{ov::element::i64, {}};
        padding_height_tensor.data<int64_t>()[0] = target_sizes.padding_height;
        image_preprocessor.set_tensor("padding_height", padding_height_tensor);

        image_preprocessor.set_tensor("attention_mask", target_sizes.attention_mask);
        
        image_preprocessor.infer();
        image_preprocessor.get_tensor("input_image_embeds").copy_to(input_image_embeds);
        image_preprocessor.get_tensor("image_attention_mask").copy_to(image_attention_mask);
        image_height = image_preprocessor.get_tensor("image_height").data<int64_t>()[0];
        image_width = image_preprocessor.get_tensor("image_width").data<int64_t>()[0];
        num_img_tokens = image_preprocessor.get_tensor("num_img_tokens").data<int64_t>()[0];
    }

    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_patch_position_ids_model.get()};
        ov::InferRequest& image_preprocessor = lock.get();
        image_preprocessor.set_tensor("input_image_embeds", input_image_embeds);
        image_preprocessor.set_tensor("image_attention_mask", image_attention_mask);
        image_preprocessor.infer();
        image_preprocessor.get_tensor("patch_position_ids").copy_to(patch_position_ids);
    }

    ov::Tensor img_features{ov::element::f32, {}};
    {
        ov::Tensor int64{ov::element::i64, patch_position_ids.get_shape()};
        std::transform(patch_position_ids.data<float>(), patch_position_ids.data<float>() + patch_position_ids.get_size(), int64.data<int64_t>(), [](float v) {return static_cast<int64_t>(v);});
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_ireq_queue_vision_encoder.get()};
        ov::InferRequest& encoder = lock.get();
        ov::Shape shape = input_image_embeds.get_shape();
        shape.erase(shape.begin());
        input_image_embeds.set_shape(shape);
        encoder.set_tensor("pixel_values", input_image_embeds);
        shape = image_attention_mask.get_shape();
        shape.erase(shape.begin());
        ov::Tensor bools{ov::element::boolean, shape};
        std::transform(image_attention_mask.data<float>(), image_attention_mask.data<float>() + image_attention_mask.get_size(), bools.data<bool>(), [](float v) {return v > 0.5f;});
        encoder.set_tensor("patch_attention_mask", bools);
        encoder.set_tensor("patch_position_ids", int64);
        encoder.infer();
        encoder.get_output_tensor().copy_to(img_features);
    }
    // Reshape img_features to add batch dimension: [?, ?, ?] -> [1, ?, ?, ?]
    ov::Shape shape = img_features.get_shape();
    shape.insert(shape.begin(), 1);
    img_features.set_shape(shape);

    ov::Tensor _1le{ov::element::f32, {}}; // l - length, e - single embedding size
    {
        ov::Tensor height{ov::element::i32, {}};
        ov::Tensor width{ov::element::i32, {}};
        ov::Tensor sub_GN{ov::element::f32, {1, 1, 1, m_vlm_config.sub_GN.size()}, m_vlm_config.sub_GN.data()};
        ov::Tensor glb_GN{ov::element::f32, {1, 1, m_vlm_config.glb_GN.size()}, m_vlm_config.glb_GN.data()};
        height.data<int32_t>()[0] = image_height;
        width.data<int32_t>()[0] = image_width;
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_separator_inserters.get()};
        ov::InferRequest& encoder = lock.get();
        encoder.set_tensor("img_features", img_features);
        encoder.set_tensor("height", height);
        encoder.set_tensor("width", width);
        encoder.set_tensor("sub_GN", sub_GN);
        encoder.set_tensor("glb_GN", glb_GN);
        encoder.infer();
        encoder.get_output_tensor().copy_to(_1le);
    }
    EncodedImage encoded_image;
    encoded_image.resized_source = _1le;
    encoded_image.images_features_projection = ov::Tensor{ov::element::f32, {}};
    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_ireq_queue_vision_projection.get()};
        ov::InferRequest& projector = lock.get();
        projector.set_input_tensor(_1le);
        projector.infer();
        projector.get_output_tensor().copy_to(encoded_image.images_features_projection);
    }
    return encoded_image;
}


InputsEmbedderPhi4MM::InputsEmbedderPhi4MM(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}


InputsEmbedderPhi4MM::InputsEmbedderPhi4MM(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}


// FIXME Copied from Phi3 (except debug tensors printing and comparing) - reuse
ov::Tensor InputsEmbedderPhi4MM::get_inputs_embeds(const std::string& image_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings) {
    size_t base_id = m_tokens_per_images.size();
    std::string prompt = phi_utils::normalize_prompt(image_prompt, base_id, images.size(), NATIVE_PATTERN, write_native);
    std::vector<ov::Tensor> images_features_proj;
    for (const ov::genai::EncodedImage& encoded_image : images) {
        images_features_proj.push_back(encoded_image.images_features_projection);
        m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
    }

    std::vector<std::variant<ov::Tensor, size_t>> new_chat_tokens;
    if (m_is_chat_conversation) {
        m_history.push_back({{"role", "user"}, {"content", std::move(image_prompt)}});
        constexpr bool add_generation_prompt = true;
        std::string new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = phi_utils::split_tokenize(new_templated_chat_history, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    } else {
        std::string templated_prompt;
        if (m_apply_chat_template) {
            ChatHistory history({{{"role", "user"}, {"content", std::move(image_prompt)}}});
            constexpr bool add_generation_prompt = true;
            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
        } else {
            templated_prompt = std::move(image_prompt);
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = phi_utils::split_tokenize(templated_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    }
    ov::Tensor new_merged_tokens = phi_utils::insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
    ov::Tensor new_tokens = update_history(new_merged_tokens);
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_tokens);

    std::vector<std::variant<ov::Tensor, size_t>> tokens = phi_utils::drop_image_placeholders(new_tokens);
    ov::Tensor inputs_embeds{ov::element::f32, {1, new_tokens.get_shape().at(1), m_vlm_config.hidden_size}};
    size_t offset = 0;
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    for (const std::variant<ov::Tensor, size_t>& chunk : tokens) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                const ov::Tensor& text_embeds = m_embedding->infer(req, chunk);
                size_t text_length = text_embeds.get_shape().at(1);
                std::copy_n(
                    text_embeds.data<float>(),
                    text_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return text_length;
            },
            [&](size_t image_id) {
                const ov::Tensor& image_embeds = images_features_proj.at(image_id - base_id);
                size_t im_length = image_embeds.get_shape().at(1);
                std::copy_n(
                    image_embeds.data<float>(),
                    image_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return im_length;
            }
        }, chunk);
    }

    if (!m_is_chat_conversation) {
        m_tokens_per_images.clear();
    }
    return inputs_embeds;
}

// FIXME Copied from Phi3 - reuse
void InputsEmbedderPhi4MM::update_chat_history(
    const std::string& decoded_results, 
    const ov::genai::GenerationStatus generation_finish_status
) {
    IInputsEmbedder::update_chat_history(decoded_results, generation_finish_status);
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL) {
        m_tokens_per_images = m_prev_tokens_per_images;
    } else {
        m_prev_tokens_per_images = m_tokens_per_images;
    }
}

// FIXME Copied from Phi3 - reuse
void InputsEmbedderPhi4MM::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_tokens_per_images.clear();
}

// FIXME Copied from Phi3 - reuse
void InputsEmbedderPhi4MM::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_tokens_per_images.clear();
}

} // namespace ov::genai
