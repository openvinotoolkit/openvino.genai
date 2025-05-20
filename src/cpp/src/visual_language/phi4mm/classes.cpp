// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/phi4mm/classes.hpp"

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

    auto t0 = make_shared<Parameter>(u8, PartialShape{-1, -1, -1});  //  -> u8[?,?,?]
    t0->output(0).get_tensor().set_names({"image"});
    auto t1 = make_shared<Constant>(f32, Shape{}, 1.0);         //  -> f32[](1.0)
    auto t2 = make_shared<Constant>(f32, Shape{}, 32.0);        //  -> f32[](32.0)
    auto t3 = make_shared<Convert>(t0, f32);                    // u8[?,?,?] -> f32[?,?,?]
    auto t4 = make_shared<Constant>(f32, Shape{1, 1, 1}, 255.0f);  //  -> f32[1,1,1]([[[255.0]]])
    auto t5 = make_shared<Divide>(t3, t4, "numpy");             // f32[?,?,?], f32[1,1,1] -> f32[?,?,?]
    auto t6 = make_shared<Constant>(f32, Shape{3, 1, 1}, -0.5f);  //  -> f32[3,1,1]([[[-0.5]], [[-0.5]], [[-0.5]]])
    auto t7 = make_shared<Add>(t5, t6);                         // f32[?,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t8 = make_shared<Constant>(f32, Shape{3, 1, 1}, 0.5f);  //  -> f32[3,1,1]([[[0.5]], [[0.5]], [[0.5]]])
    auto t9 = make_shared<Divide>(t7, t8, "numpy");             // f32[3,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t10 = make_shared<ShapeOf>(t9);                        // f32[3,?,?] -> i64[3]
    auto t11 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t12 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t13 = make_shared<Gather>(t10, t11, t12);              // i64[3], i64[], i64[] -> i64[]
    auto t14 = make_shared<Convert>(t13, f32);                  // i64[] -> f32[]
    auto t15 = make_shared<Constant>(f32, Shape{}, 448.0);      //  -> f32[](448.0)
    auto t16 = make_shared<Divide>(t14, t15, "numpy");          // f32[], f32[] -> f32[]
    auto t17 = make_shared<Ceiling>(t16);                       // f32[] -> f32[]
    auto t18 = make_shared<Multiply>(t2, t17, "numpy");         // f32[], f32[] -> f32[]
    auto t19 = make_shared<Convert>(t18, i32);                  // f32[] -> i32[]
    auto t20 = make_shared<Convert>(t19, i64);                  // i32[] -> i64[]
    auto t21 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t22 = make_shared<Unsqueeze>(t20, t21);                // i64[], i32[] -> i64[1]
    auto t23 = make_shared<Constant>(i64, Shape{}, 2);          //  -> i64[](2)
    auto t24 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t25 = make_shared<Gather>(t10, t23, t24);              // i64[3], i64[], i64[] -> i64[]
    auto t26 = make_shared<Convert>(t25, f32);                  // i64[] -> f32[]
    auto t27 = make_shared<Divide>(t26, t15, "numpy");          // f32[], f32[] -> f32[]
    auto t28 = make_shared<Ceiling>(t27);                       // f32[] -> f32[]
    auto t29 = make_shared<Multiply>(t2, t28, "numpy");         // f32[], f32[] -> f32[]
    auto t30 = make_shared<Convert>(t29, i32);                  // f32[] -> i32[]
    auto t31 = make_shared<Convert>(t30, i64);                  // i32[] -> i64[]
    auto t32 = make_shared<Unsqueeze>(t31, t21);                // i64[], i32[] -> i64[1]
    auto t33 = make_shared<Concat>(NodeVector{t22, t32}, 0);    // i64[1], i64[1] -> i64[2]
    auto t34 = make_shared<Broadcast>(t1, t33);                 // f32[], i64[2] -> f32[?,?]
    auto t35 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t36 = make_shared<Reshape>(t34, t35, false);           // f32[?,?], i64[1] -> f32[?]
    auto t37 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t38 = make_shared<ShapeOf>(t34);                       // f32[?,?] -> i64[2]
    auto t39 = make_shared<ReduceProd>(t38, t37);               // i64[2], i64[] -> i64[]
    auto t40 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t41 = make_shared<Range>(t37, t39, t40, i64);          // i64[], i64[], i64[] -> i64[?]
    auto t42 = make_shared<Reshape>(t41, t38, false);           // i64[?], i64[2] -> i64[?,?]
    auto t43 = make_shared<Multiply>(t15, t28, "numpy");        // f32[], f32[] -> f32[]
    auto t44 = make_shared<Divide>(t43, t26, "numpy");          // f32[], f32[] -> f32[]
    auto t45 = make_shared<Multiply>(t15, t17, "numpy");        // f32[], f32[] -> f32[]
    auto t46 = make_shared<Divide>(t45, t14, "numpy");          // f32[], f32[] -> f32[]
    auto t47 = make_shared<Less>(t44, t46);                     // f32[], f32[] -> boolean[]
    auto t48 = make_shared<Convert>(t47, i64);                  // boolean[] -> i64[]
    auto t49 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t50 = make_shared<Multiply>(t48, t49, "numpy");        // i64[], i64[] -> i64[]
    auto t51 = make_shared<GreaterEqual>(t44, t46);             // f32[], f32[] -> boolean[]
    auto t52 = make_shared<Convert>(t51, i64);                  // boolean[] -> i64[]
    auto t53 = make_shared<Multiply>(t26, t46, "numpy");        // f32[], f32[] -> f32[]
    auto t54 = make_shared<Convert>(t53, i32);                  // f32[] -> i32[]
    auto t55 = make_shared<Convert>(t54, f32);                  // i32[] -> f32[]
    auto t56 = make_shared<Subtract>(t43, t55);                 // f32[], f32[] -> f32[]
    auto t57 = make_shared<Convert>(t56, i32);                  // f32[] -> i32[]
    auto t58 = make_shared<Convert>(t57, i64);                  // i32[] -> i64[]
    auto t59 = make_shared<Multiply>(t52, t58, "numpy");        // i64[], i64[] -> i64[]
    auto t60 = make_shared<Add>(t50, t59);                      // i64[], i64[] -> i64[]
    auto t61 = make_shared<Convert>(t60, i32);                  // i64[] -> i32[]
    auto t62 = make_shared<Convert>(t61, f32);                  // i32[] -> f32[]
    auto t63 = make_shared<Constant>(f32, Shape{}, 14.0);       //  -> f32[](14.0)
    auto t64 = make_shared<Divide>(t62, t63, "numpy");          // f32[], f32[] -> f32[]
    auto t65 = make_shared<Convert>(t64, i32);                  // f32[] -> i32[]
    auto t66 = make_shared<Subtract>(t30, t65);                 // i32[], i32[] -> i32[]
    auto t67 = make_shared<Convert>(t66, i64);                  // i32[] -> i64[]
    auto t68 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t69 = make_shared<Reshape>(t67, t68, false);           // i64[], i64[1] -> i64[1]
    auto t70 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t71 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t72 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t73 = make_shared<Slice>(t42, t69, t70, t71, t72);     // i64[?,?], i64[1], i64[1], i64[1], i64[1] -> i64[?,?]
    auto t74 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{-1, 1});  //  -> i64[2]([-1, 1])
    auto t75 = make_shared<Reshape>(t73, t74, false);           // i64[?,?], i64[2] -> i64[?,1]
    auto t76 = make_shared<Constant>(f32, Shape{}, 0.0);        //  -> f32[](0.0)
    auto t77 = make_shared<Slice>(t34, t69, t70, t71, t72);     // f32[?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?]
    auto t78 = make_shared<ShapeOf>(t77);                       // f32[?,?] -> i32[2]
    auto t79 = make_shared<Broadcast>(t76, t78);                // f32[], i32[2] -> f32[?,?]
    auto t80 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t81 = make_shared<Reshape>(t79, t80, false);           // f32[?,?], i64[1] -> f32[?]
    auto t82 = make_shared<ScatterNDUpdate>(t36, t75, t81);     // f32[?], i64[?,1], f32[?] -> f32[?]
    auto t83 = make_shared<Reshape>(t82, t38, false);           // f32[?], i64[2] -> f32[?,?]
    auto t84 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t85 = make_shared<Reshape>(t83, t84, false);           // f32[?,?], i64[1] -> f32[?]
    auto t86 = make_shared<Multiply>(t14, t44, "numpy");        // f32[], f32[] -> f32[]
    auto t87 = make_shared<Convert>(t86, i32);                  // f32[] -> i32[]
    auto t88 = make_shared<Convert>(t87, f32);                  // i32[] -> f32[]
    auto t89 = make_shared<Subtract>(t45, t88);                 // f32[], f32[] -> f32[]
    auto t90 = make_shared<Convert>(t89, i32);                  // f32[] -> i32[]
    auto t91 = make_shared<Convert>(t90, i64);                  // i32[] -> i64[]
    auto t92 = make_shared<Multiply>(t48, t91, "numpy");        // i64[], i64[] -> i64[]
    auto t93 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t94 = make_shared<Multiply>(t52, t93, "numpy");        // i64[], i64[] -> i64[]
    auto t95 = make_shared<Add>(t92, t94);                      // i64[], i64[] -> i64[]
    auto t96 = make_shared<Convert>(t95, i32);                  // i64[] -> i32[]
    auto t97 = make_shared<Convert>(t96, f32);                  // i32[] -> f32[]
    auto t98 = make_shared<Constant>(f32, Shape{}, 14.0);       //  -> f32[](14.0)
    auto t99 = make_shared<Divide>(t97, t98, "numpy");          // f32[], f32[] -> f32[]
    auto t100 = make_shared<Convert>(t99, i32);                 // f32[] -> i32[]
    auto t101 = make_shared<Subtract>(t19, t100);               // i32[], i32[] -> i32[]
    auto t102 = make_shared<Convert>(t101, i64);                // i32[] -> i64[]
    auto t103 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t104 = make_shared<Reshape>(t102, t103, false);        // i64[], i64[1] -> i64[1]
    auto t105 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t106 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t107 = make_shared<Constant>(i64, Shape{1}, 0);        //  -> i64[1]([0])
    auto t108 = make_shared<Slice>(t42, t104, t105, t106, t107);  // i64[?,?], i64[1], i64[1], i64[1], i64[1] -> i64[?,?]
    auto t109 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{-1, 1});  //  -> i64[2]([-1, 1])
    auto t110 = make_shared<Reshape>(t108, t109, false);        // i64[?,?], i64[2] -> i64[?,1]
    auto t111 = make_shared<Constant>(f32, Shape{}, 0.0);       //  -> f32[](0.0)
    auto t112 = make_shared<Slice>(t83, t104, t105, t106, t107);  // f32[?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?]
    auto t113 = make_shared<ShapeOf>(t112);                     // f32[?,?] -> i32[2]
    auto t114 = make_shared<Broadcast>(t111, t113);             // f32[], i32[2] -> f32[?,?]
    auto t115 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t116 = make_shared<Reshape>(t114, t115, false);        // f32[?,?], i64[1] -> f32[?]
    auto t117 = make_shared<ScatterNDUpdate>(t85, t110, t116);  // f32[?], i64[?,1], f32[?] -> f32[?]
    auto t118 = make_shared<Reshape>(t117, t38, false);         // f32[?], i64[2] -> f32[?,?]
    auto t119 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t120 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t121 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t122 = make_shared<Gather>(t38, t120, t121);           // i64[2], i64[], i64[] -> i64[]
    auto t123 = make_shared<Constant>(i64, Shape{}, 32);        //  -> i64[](32)
    auto t124 = make_shared<Divide>(t122, t123, "numpy");       // i64[], i64[] -> i64[]
    auto t125 = make_shared<Floor>(t124);                       // i64[] -> i64[]
    auto t126 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t127 = make_shared<Unsqueeze>(t125, t126);             // i64[], i32[] -> i64[1]
    auto t128 = make_shared<Constant>(i64, Shape{1}, 32);       //  -> i64[1]([32])
    auto t129 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t130 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t131 = make_shared<Gather>(t38, t129, t130);           // i64[2], i64[], i32[] -> i64[]
    auto t132 = make_shared<Divide>(t131, t123, "numpy");       // i64[], i64[] -> i64[]
    auto t133 = make_shared<Floor>(t132);                       // i64[] -> i64[]
    auto t134 = make_shared<Unsqueeze>(t133, t126);             // i64[], i32[] -> i64[1]
    auto t135 = make_shared<Constant>(i64, Shape{1}, 32);       //  -> i64[1]([32])
    auto t136 = make_shared<Concat>(NodeVector{t119, t127, t128, t134, t135}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t137 = make_shared<Reshape>(t118, t136, false);        // f32[?,?], i64[5] -> f32[?,?,?,?,?]
    auto t138 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t139 = make_shared<Transpose>(t137, t138);             // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t140 = make_shared<Constant>(i64, Shape{3}, vector<int64_t>{-1, 32, 32});  //  -> i64[3]([-1, 32, 32])
    auto t141 = make_shared<Reshape>(t139, t140, false);        // f32[?,?,?,?,?], i64[3] -> f32[?,32,32]
    auto t142 = make_shared<Constant>(i64, Shape{2}, 0);        //  -> i64[2]([0, 0])
    auto t143 = make_shared<Constant>(i64, Shape{2}, 9223372036854775807);  //  -> i64[2]([9223372036854775807, 9223372036854775807])
    auto t144 = make_shared<Constant>(i64, Shape{2}, 2);        //  -> i64[2]([2, 2])
    auto t145 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 2});  //  -> i64[2]([1, 2])
    auto t146 = make_shared<Slice>(t141, t142, t143, t144, t145);  // f32[?,32,32], i64[2], i64[2], i64[2], i64[2] -> f32[?,16,16]
    auto t147 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t148 = make_shared<Constant>(i64, Shape{1}, 16);       //  -> i64[1]([16])
    auto t149 = make_shared<Constant>(i64, Shape{1}, 16);       //  -> i64[1]([16])
    auto t150 = make_shared<Concat>(NodeVector{t147, t127, t134, t148, t149}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t151 = make_shared<Reshape>(t146, t150, false);        // f32[?,16,16], i64[5] -> f32[?,?,?,?,?]
    auto t152 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t153 = make_shared<Transpose>(t151, t152);             // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t154 = make_shared<ShapeOf>(t153);                     // f32[?,?,?,?,?] -> i64[5]
    auto t155 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t156 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t157 = make_shared<Gather>(t154, t155, t156);          // i64[5], i64[1], i64[] -> i64[1]
    auto t158 = make_shared<Constant>(i64, Shape{1}, 2);        //  -> i64[1]([2])
    auto t159 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t160 = make_shared<Gather>(t154, t158, t159);          // i64[5], i64[1], i64[] -> i64[1]
    auto t161 = make_shared<Multiply>(t157, t160, "numpy");     // i64[1], i64[1] -> i64[1]
    auto t162 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t163 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t164 = make_shared<Gather>(t154, t162, t163);          // i64[5], i64[1], i64[] -> i64[1]
    auto t165 = make_shared<Constant>(i64, Shape{1}, 4);        //  -> i64[1]([4])
    auto t166 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t167 = make_shared<Gather>(t154, t165, t166);          // i64[5], i64[1], i64[] -> i64[1]
    auto t168 = make_shared<Multiply>(t164, t167, "numpy");     // i64[1], i64[1] -> i64[1]
    auto t169 = make_shared<Concat>(NodeVector{t161, t168}, 0);  // i64[1], i64[1] -> i64[2]
    auto t170 = make_shared<Reshape>(t153, t169, false);        // f32[?,?,?,?,?], i64[2] -> f32[?,?]
    auto t171 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{0, 1});  //  -> i32[2]([0, 1])
    auto t172 = make_shared<ReduceSum>(t170, t171);             // f32[?,?], i32[2] -> f32[]
    auto t173 = make_shared<Constant>(f32, Shape{}, 257.0);     //  -> f32[](257.0)
    auto t174 = make_shared<Add>(t172, t173);                   // f32[], f32[] -> f32[]
    auto t175 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t176 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t177 = make_shared<Gather>(t170, t175, t176);          // f32[?,?], i64[], i64[] -> f32[?]
    auto t178 = make_shared<Constant>(i32, Shape{1}, 0);        //  -> i32[1]([0])
    auto t179 = make_shared<ReduceSum>(t177, t178);             // f32[?], i32[1] -> f32[]
    auto t180 = make_shared<Add>(t174, t179);                   // f32[], f32[] -> f32[]
    auto t181 = make_shared<Constant>(f32, Shape{}, 16.0);      //  -> f32[](16.0)
    auto t182 = make_shared<Add>(t180, t181);                   // f32[], f32[] -> f32[]
    auto t183 = make_shared<Constant>(f32, Shape{1, 32, 32}, 1.0f);  //  -> f32[1,32,32]
    auto t184 = make_shared<Concat>(NodeVector{t183, t141}, 0);  // f32[1,32,32], f32[?,32,32] -> f32[1..,32,32]
    auto t185 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t186 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t187 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 2});  //  -> i64[2]([1, 2])
    auto t188 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t189 = make_shared<Gather>(t10, t187, t188);           // i64[3], i64[2], i64[] -> i64[2]
    auto t190 = make_shared<Concat>(NodeVector{t185, t186, t189}, 0);  // i64[1], i64[1], i64[2] -> i64[4]
    auto t191 = make_shared<Reshape>(t9, t190, false);          // f32[3,?,?], i64[4] -> f32[?,3,?,?]
    auto t192 = make_shared<Convert>(t86, i64);                 // f32[] -> i64[]
    auto t193 = make_shared<Multiply>(t48, t192, "numpy");      // i64[], i64[] -> i64[]
    auto t194 = make_shared<Convert>(t45, i64);                 // f32[] -> i64[]
    auto t195 = make_shared<Multiply>(t52, t194, "numpy");      // i64[], i64[] -> i64[]
    auto t196 = make_shared<Add>(t193, t195);                   // i64[], i64[] -> i64[]
    auto t197 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t198 = make_shared<Unsqueeze>(t196, t197);             // i64[], i32[] -> i64[1]
    auto t199 = make_shared<Convert>(t43, i64);                 // f32[] -> i64[]
    auto t200 = make_shared<Multiply>(t48, t199, "numpy");      // i64[], i64[] -> i64[]
    auto t201 = make_shared<Convert>(t53, i64);                 // f32[] -> i64[]
    auto t202 = make_shared<Multiply>(t52, t201, "numpy");      // i64[], i64[] -> i64[]
    auto t203 = make_shared<Add>(t200, t202);                   // i64[], i64[] -> i64[]
    auto t204 = make_shared<Unsqueeze>(t203, t197);             // i64[], i32[] -> i64[1]
    auto t205 = make_shared<Concat>(NodeVector{t198, t204}, 0);  // i64[1], i64[1] -> i64[2]
    auto t206 = make_shared<Convert>(t205, i32);                // i64[2] -> i32[2]
    auto t207 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t208_attrs{Interpolate::InterpolateMode::BILINEAR_PILLOW, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t208 = make_shared<Interpolate>(t191, t206, t207, t208_attrs);  // f32[?,3,?,?], i32[2], i32[2] -> f32[?,3,?,?]
    auto t209 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t210 = make_shared<Concat>(NodeVector{t209, t198, t204}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t211 = make_shared<Reshape>(t208, t210, false);        // f32[?,3,?,?], i64[3] -> f32[?,?,?]
    auto t212 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t213 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t214 = make_shared<Convert>(t61, i64);                 // i32[] -> i64[]
    auto t215 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t216 = make_shared<Unsqueeze>(t214, t215);             // i64[], i32[] -> i64[1]
    auto t217 = make_shared<Concat>(NodeVector{t213, t198, t216}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t218 = make_shared<Broadcast>(t212, t217);             // f32[], i64[3] -> f32[3,?,?]
    auto t219 = make_shared<Concat>(NodeVector{t211, t218}, 2);  // f32[?,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t220 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t221 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t222 = make_shared<Convert>(t96, i64);                 // i32[] -> i64[]
    auto t223 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t224 = make_shared<Unsqueeze>(t222, t223);             // i64[], i32[] -> i64[1]
    auto t225 = make_shared<Add>(t214, t203);                   // i64[], i64[] -> i64[]
    auto t226 = make_shared<Unsqueeze>(t225, t223);             // i64[], i32[] -> i64[1]
    auto t227 = make_shared<Concat>(NodeVector{t221, t224, t226}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t228 = make_shared<Broadcast>(t220, t227);             // f32[], i64[3] -> f32[3,?,?]
    auto t229 = make_shared<Concat>(NodeVector{t219, t228}, 1);  // f32[3,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t230 = make_shared<ShapeOf>(t229);                     // f32[3,?,?] -> i64[3]
    auto t231 = make_shared<Constant>(i64, Shape{}, 2);         //  -> i64[](2)
    auto t232 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t233 = make_shared<Gather>(t230, t231, t232);          // i64[3], i64[], i32[] -> i64[]
    auto t234 = make_shared<ShapeOf>(t219);                     // f32[3,?,?] -> i64[3]
    auto t235 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t236 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t237 = make_shared<Gather>(t234, t235, t236);          // i64[3], i64[], i64[] -> i64[]
    auto t238 = make_shared<ShapeOf>(t228);                     // f32[3,?,?] -> i64[3]
    auto t239 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t240 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t241 = make_shared<Gather>(t238, t239, t240);          // i64[3], i64[], i64[] -> i64[]
    auto t242 = make_shared<Add>(t237, t241);                   // i64[], i64[] -> i64[]
    auto t243 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t244 = make_shared<Unsqueeze>(t229, t243);             // f32[3,?,?], i64[] -> f32[1,3,?,?]
    auto t245 = make_shared<Constant>(i32, Shape{2}, 448);      //  -> i32[2]([448, 448])
    auto t246 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t247_attrs{Interpolate::InterpolateMode::CUBIC, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t247 = make_shared<Interpolate>(t244, t245, t246, t247_attrs);  // f32[1,3,?,?], i32[2], i32[2] -> f32[1,3,448,448]
    auto t248 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t249 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t250 = make_shared<Constant>(i64, Shape{}, 448);       //  -> i64[](448)
    auto t251 = make_shared<Divide>(t242, t250, "numpy");       // i64[], i64[] -> i64[]
    auto t252 = make_shared<Floor>(t251);                       // i64[] -> i64[]
    auto t253 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t254 = make_shared<Unsqueeze>(t252, t253);             // i64[], i32[] -> i64[1]
    auto t255 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t256 = make_shared<Divide>(t233, t250, "numpy");       // i64[], i64[] -> i64[]
    auto t257 = make_shared<Floor>(t256);                       // i64[] -> i64[]
    auto t258 = make_shared<Unsqueeze>(t257, t253);             // i64[], i32[] -> i64[1]
    auto t259 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t260 = make_shared<Concat>(NodeVector{t248, t249, t254, t255, t258, t259}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[6]
    auto t261 = make_shared<Reshape>(t229, t260, false);        // f32[3,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t262 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 2, 4, 1, 3, 5});  //  -> i64[6]([0, 2, 4, 1, 3, 5])
    auto t263 = make_shared<Transpose>(t261, t262);             // f32[?,?,?,?,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t264 = make_shared<Constant>(i64, Shape{4}, vector<int64_t>{-1, 3, 448, 448});  //  -> i64[4]([-1, 3, 448, 448])
    auto t265 = make_shared<Reshape>(t263, t264, false);        // f32[?,?,?,?,?,?], i64[4] -> f32[?,3,448,448]
    auto t266 = make_shared<Concat>(NodeVector{t247, t265}, 0);  // f32[1,3,448,448], f32[?,3,448,448] -> f32[1..,3,448,448]
    auto t267 = make_shared<Result>(t182);                      // f32[] -> f32[]
    t267->output(0).get_tensor().set_names({"477", "num_img_tokens"});
    auto t268 = make_shared<Result>(t184);                      // f32[1..,32,32] -> f32[1..,32,32]
    t268->output(0).get_tensor().set_names({"image_attention_mask"});
    auto t269 = make_shared<Result>(t233);                      // i64[] -> i64[]
    t269->output(0).get_tensor().set_names({"341", "image_width"});
    auto t270 = make_shared<Result>(t242);                      // i64[] -> i64[]
    t270->output(0).get_tensor().set_names({"338", "image_height"});
    auto t271 = make_shared<Result>(t266);                      // f32[1..,3,448,448] -> f32[1..,3,448,448]
    t271->output(0).get_tensor().set_names({"input_image_embeds"});

    ResultVector results{t271, t270, t269, t268, t267};
    SinkVector sinks{};
    ParameterVector parameters{t0};
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
    t1->output(0).get_tensor().set_names({"base_feat_size"});
    auto t2 = make_shared<Parameter>(i32, PartialShape{});      //  -> i32[]
    t2->output(0).get_tensor().set_names({"embedding_len"});
    auto t3 = make_shared<Parameter>(i32, PartialShape{});      //  -> i32[]
    t3->output(0).get_tensor().set_names({"height"});
    auto t4 = make_shared<Parameter>(i32, PartialShape{});      //  -> i32[]
    t4->output(0).get_tensor().set_names({"width"});
    auto t5 = make_shared<Parameter>(f32, PartialShape{1, 1, 1, -1});  //  -> f32[1,1,1,?]
    t5->output(0).get_tensor().set_names({"sub_GN"});
    auto t6 = make_shared<Parameter>(f32, PartialShape{1, 1, -1});  //  -> f32[1,1,?]
    t6->output(0).get_tensor().set_names({"glb_GN"});
    auto t7 = make_shared<Constant>(i64, Shape{1}, 1);          //  -> i64[1]([1])
    auto t8 = make_shared<Constant>(i64, Shape{1}, -1);         //  -> i64[1]([-1])
    auto t9 = make_shared<Constant>(i32, Shape{}, 2);           //  -> i32[](2)
    auto t10 = make_shared<Power>(t1, t9, "numpy");             // i32[], i32[] -> i32[]
    auto t11 = make_shared<Convert>(t10, i64);                  // i32[] -> i64[]
    auto t12 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t13 = make_shared<Unsqueeze>(t11, t12);                // i64[], i32[] -> i64[1]
    auto t14 = make_shared<Convert>(t2, i64);                   // i32[] -> i64[]
    auto t15 = make_shared<Unsqueeze>(t14, t12);                // i64[], i32[] -> i64[1]
    auto t16 = make_shared<Concat>(NodeVector{t7, t8, t13, t15}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t17 = make_shared<Reshape>(t0, t16, false);            // f32[1,?,?,?], i64[4] -> f32[?,?,?,?]
    auto t18 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t19 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t20 = make_shared<Gather>(t17, t18, t19);              // f32[?,?,?,?], i64[], i64[] -> f32[?,?,?]
    auto t21 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t22 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t23 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t24 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t25 = make_shared<Slice>(t20, t21, t22, t23, t24);     // f32[?,?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?,?]
    auto t26 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t27 = make_shared<Convert>(t3, i64);                   // i32[] -> i64[]
    auto t28 = make_shared<Constant>(i64, Shape{}, 448);        //  -> i64[](448)
    auto t29 = make_shared<Divide>(t27, t28, "numpy");          // i64[], i64[] -> i64[]
    auto t30 = make_shared<Floor>(t29);                         // i64[] -> i64[]
    auto t31 = make_shared<Convert>(t4, i64);                   // i32[] -> i64[]
    auto t32 = make_shared<Constant>(i64, Shape{}, 448);        //  -> i64[](448)
    auto t33 = make_shared<Divide>(t31, t32, "numpy");          // i64[], i64[] -> i64[]
    auto t34 = make_shared<Floor>(t33);                         // i64[] -> i64[]
    auto t35 = make_shared<Multiply>(t30, t34, "numpy");        // i64[], i64[] -> i64[]
    auto t36 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t37 = make_shared<Reshape>(t35, t36, false);           // i64[], i64[1] -> i64[1]
    auto t38 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t39 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t40 = make_shared<Slice>(t25, t26, t37, t38, t39);     // f32[?,?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?,?]
    auto t41 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t42 = make_shared<Unsqueeze>(t30, t41);                // i64[], i32[] -> i64[1]
    auto t43 = make_shared<Unsqueeze>(t34, t41);                // i64[], i32[] -> i64[1]
    auto t44 = make_shared<Convert>(t1, i64);                   // i32[] -> i64[]
    auto t45 = make_shared<Unsqueeze>(t44, t41);                // i64[], i32[] -> i64[1]
    auto t46 = make_shared<Concat>(NodeVector{t42, t43, t45, t45, t15}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t47 = make_shared<Reshape>(t40, t46, false);           // f32[?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t48 = make_shared<Constant>(i32, Shape{5}, vector<int32_t>{0, 2, 1, 3, 4});  //  -> i32[5]([0, 2, 1, 3, 4])
    auto t49 = make_shared<Transpose>(t47, t48);                // f32[?,?,?,?,?], i32[5] -> f32[?,?,?,?,?]
    auto t50 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t51 = make_shared<Multiply>(t30, t44, "numpy");        // i64[], i64[] -> i64[]
    auto t52 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t53 = make_shared<Unsqueeze>(t51, t52);                // i64[], i32[] -> i64[1]
    auto t54 = make_shared<Multiply>(t34, t44, "numpy");        // i64[], i64[] -> i64[]
    auto t55 = make_shared<Unsqueeze>(t54, t52);                // i64[], i32[] -> i64[1]
    auto t56 = make_shared<Concat>(NodeVector{t50, t53, t55, t15}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t57 = make_shared<Reshape>(t49, t56, false);           // f32[?,?,?,?,?], i64[4] -> f32[?,?,?,?]
    auto t58 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t59 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t60 = make_shared<Reshape>(t51, t59, false);           // i64[], i64[1] -> i64[1]
    auto t61 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t62 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t63 = make_shared<Concat>(NodeVector{t58, t60, t61, t62}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t64 = make_shared<Tile>(t5, t63);                      // f32[1,1,1,?], i64[4] -> f32[?,?,?,?]
    auto t65 = make_shared<Concat>(NodeVector{t57, t64}, 2);    // f32[?,?,?,?], f32[?,?,?,?] -> f32[?,?,?,?]
    auto t66 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t67 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t68 = make_shared<Concat>(NodeVector{t66, t67, t15}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t69 = make_shared<Reshape>(t65, t68, false);           // f32[?,?,?,?], i64[3] -> f32[?,?,?]
    auto t70 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t71 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t72 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t73 = make_shared<Constant>(i64, Shape{1}, 0);         //  -> i64[1]([0])
    auto t74 = make_shared<Slice>(t20, t70, t71, t72, t73);     // f32[?,?,?], i64[1], i64[1], i64[1], i64[1] -> f32[..1,?,?]
    auto t75 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t76 = make_shared<Concat>(NodeVector{t75, t45, t45, t15}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t77 = make_shared<Reshape>(t74, t76, false);           // f32[..1,?,?], i64[4] -> f32[?,?,?,?]
    auto t78 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t79 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t80 = make_shared<Reshape>(t44, t79, false);           // i64[], i64[1] -> i64[1]
    auto t81 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t82 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t83 = make_shared<Concat>(NodeVector{t78, t80, t81, t82}, 0);  // i64[1], i64[1], i64[1], i64[1] -> i64[4]
    auto t84 = make_shared<Tile>(t5, t83);                      // f32[1,1,1,?], i64[4] -> f32[?,?,?,?]
    auto t85 = make_shared<Concat>(NodeVector{t77, t84}, 2);    // f32[?,?,?,?], f32[?,?,?,?] -> f32[?,?,?,?]
    auto t86 = make_shared<Reshape>(t85, t68, false);           // f32[?,?,?,?], i64[3] -> f32[?,?,?]
    auto t87 = make_shared<Concat>(NodeVector{t69, t6, t86}, 1);  // f32[?,?,?], f32[1,1,?], f32[?,?,?] -> f32[1,1..,?]
    auto t88 = make_shared<Result>(t87);                        // f32[1,1..,?] -> f32[1,1..,?]
    t88->output(0).get_tensor().set_names({});

    ResultVector results{t88};
    SinkVector sinks{};
    ParameterVector parameters{t0, t1, t2, t3, t4, t5, t6};
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

} // namespace

namespace ov::genai {

VisionEncoderPhi4MM::VisionEncoderPhi4MM(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties
) :
VisionEncoder(model_dir, device, properties),
m_image_preprocessors{create_image_preprocessors()},
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
) : VisionEncoder(models_map, config_dir_path, device, properties), m_image_preprocessors{create_image_preprocessors()} {
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

// infer preprocess:
//             "input_image_embeds"
//             "image_sizes"
//             "image_attention_mask"
//             "num_img_tokens"
//             "patch_position_ids"
// encoder()
// my C++
// vision_projection

EncodedImage VisionEncoderPhi4MM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);
    ov::Tensor input_image_embeds, image_attention_mask, patch_position_ids;
    int32_t image_height = 0, image_width = 0, num_img_tokens = 0;
    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_image_preprocessors.get()};
        ov::InferRequest& image_preprocessor = lock.get();
        ov::Shape image_shape = image.get_shape();
        std::cout << image_shape << '\n';
        image_shape.erase(image_shape.begin());
        image_preprocessor.set_input_tensor(ov::Tensor{image.get_element_type(), image_shape, image.data()});
        image_preprocessor.infer();
        image_preprocessor.get_tensor("input_image_embeds").copy_to(input_image_embeds);
        image_preprocessor.get_tensor("image_attention_mask").copy_to(image_attention_mask);
        // image_preprocessor.get_tensor("patch_position_ids").copy_to(patch_position_ids);
        image_height = image_preprocessor.get_tensor("image_height").data<int32_t>()[0];
        image_width = image_preprocessor.get_tensor("image_width").data<int32_t>()[0];
        num_img_tokens = image_preprocessor.get_tensor("num_img_tokens").data<int32_t>()[0];
    }
    std::cout << "AAAAAAAAAAAAAAAAAAAaa\n";
    ov::Tensor img_features;
    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_ireq_queue_vision_encoder.get()};
        ov::InferRequest& encoder = lock.get();
        encoder.set_tensor("pixel_values", input_image_embeds);
        encoder.set_tensor("image_attention_mask", image_attention_mask);
        encoder.set_tensor("patch_position_ids", patch_position_ids);
        encoder.infer();
        encoder.get_output_tensor().copy_to(img_features);
    }
    std::cout << "BBBBBBBBBBBBBBBBBBBBBBb\n";
    ov::Tensor _1le; // l - length, e - single embedding size
    {
        ov::Tensor base_feat_size{ov::element::i32, {0}};
        ov::Tensor embedding_len{ov::element::i32, {0}};
        ov::Tensor height{ov::element::i32, {0}};
        ov::Tensor width{ov::element::i32, {0}};
        ov::Tensor sub_GN{ov::element::f32, {1, 1, 1, m_vlm_config.sub_GN.size()}, m_vlm_config.sub_GN.data()};
        ov::Tensor glb_GN{ov::element::f32, {1, 1, m_vlm_config.glb_GN.size()}, m_vlm_config.glb_GN.data()};
        base_feat_size.data<int32_t>()[0] = std::sqrt(img_features.get_shape()[2]);
        std::cout << img_features.get_shape() << '\n';
        embedding_len.data<int32_t>()[0] = img_features.get_shape()[3];
        height.data<int32_t>()[0] = image_height;
        width.data<int32_t>()[0] = image_width;
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_separator_inserters.get()};
        ov::InferRequest& encoder = lock.get();
        encoder.set_tensor("img_features", img_features);
        encoder.set_tensor("base_feat_size", base_feat_size);
        encoder.set_tensor("embedding_len", embedding_len);
        encoder.set_tensor("height", height);
        encoder.set_tensor("width", width);
        encoder.set_tensor("sub_GN", sub_GN);
        encoder.set_tensor("glb_GN", glb_GN);
        encoder.infer();
        encoder.get_output_tensor().copy_to(_1le);
    }
    std::cout << "CCCCCCCCCCCCCCCCCCCCC\n";
    // CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    // ov::InferRequest& encoder = infer_request_guard.get();

    // Create attention mask for image patches
    // int height = image.get_shape()[1];
    // int width = image.get_shape()[2];
    // ov::Tensor patch_attention_mask = ov::Tensor{ov::element::boolean, {1, height, width}};
    // std::fill_n(patch_attention_mask.data<bool>(), patch_attention_mask.get_size(), true);

    // // Get position IDs for the image
    // ov::Tensor patch_position_ids = get_vision_position_ids(
    //     image, 
    //     patch_attention_mask, 
    //     config.patch_size, 
    //     config.vision_config_image_size / config.patch_size
    // );
    
    // encoder.set_input_tensor("pixel_values", image);
    // encoder.set_input_tensor("patch_attention_mask", patch_attention_mask);
    // encoder.set_input_tensor("patch_position_ids", patch_position_ids);
    // encoder.infer();
    // ov::Tensor vision_features = encoder.get_output_tensor();

    // // Create encoded image result
    // EncodedImage encoded_image;
    // encoded_image.resized_source = vision_features;
    // encoded_image.resized_source_size = {
    //     static_cast<size_t>(height / config.patch_size),
    //     static_cast<size_t>(width / config.patch_size)
    // };
    
    // CircularBufferQueueElementGuard<ov::InferRequest> vision_projection_ireq_guard(this->m_ireq_queue_vision_projection.get());
    // ov::InferRequest& vision_projection = vision_projection_ireq_guard.get();
    // vision_projection.set_input_tensor(vision_features);
    // vision_projection.infer();
    // encoded_image.images_features_projection = vision_projection.get_output_tensor();
    
    // return encoded_image;


    // Using mocked tensors
    EncodedImage encoded_image;

    // ov::Tensor img_features = read_tensor_from_file("./temp/tensors/phi4mm/img_features.bin");

    // encoded_image.resized_source = img_features;
    
    // ov::Shape shape = img_features.get_shape();
    // encoded_image.resized_source_size = {
    //     static_cast<size_t>(shape[1] / m_processor_config.patch_size),
    //     static_cast<size_t>(shape[2] / m_processor_config.patch_size)
    // };
    
    // encoded_image.original_image_size = {
    //     static_cast<size_t>(image.get_shape()[2]),
    //     static_cast<size_t>(image.get_shape()[1])
    // };

    // ov::Tensor img_feature_proj = read_tensor_from_file("./temp/tensors/phi4mm/img_feature_proj.bin");
    
    // encoded_image.images_features_projection = img_feature_proj;
    
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
ov::Tensor InputsEmbedderPhi4MM::get_inputs_embeds(
    const std::string& prompt, 
    const std::vector<ov::genai::EncodedImage>& images, 
    ov::genai::VLMPerfMetrics& metrics, 
    bool recalculate_merged_embeddings
) {
    size_t base_id = m_tokens_per_images.size();
    std::string image_prompt = util::normalize_prompt(prompt, base_id, images.size());

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
        new_chat_tokens = util::split_tokenize(new_templated_chat_history, m_tokenizer);
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
        new_chat_tokens = util::split_tokenize(templated_prompt, m_tokenizer);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    }
    
    ov::Tensor new_merged_tokens = util::insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
    ov::Tensor new_tokens = update_history(new_merged_tokens);
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_tokens);

    std::vector<std::variant<ov::Tensor, size_t>> tokens = util::drop_image_placeholders(new_tokens);
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

// FIXME Copied from Phi3 - reuse
bool InputsEmbedderPhi4MM::prompt_has_image_tag(const std::string& prompt) const {
    return IInputsEmbedder::prompt_has_image_tag(prompt) || std::regex_search(prompt, NATIVE_PATTERN);
}

} // namespace ov::genai
