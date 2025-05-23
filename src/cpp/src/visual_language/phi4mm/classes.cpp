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

    auto t0 = make_shared<Parameter>(u8, PartialShape{-1, -1, -1, -1});  //  -> u8[?,?,?,?]
    t0->output(0).get_tensor().set_names({"image"});
    auto t1 = make_shared<Constant>(f32, Shape{}, 1.0);         //  -> f32[](1.0)
    auto t2 = make_shared<Constant>(f32, Shape{}, 32.0);        //  -> f32[](32.0)
    auto t3 = make_shared<Constant>(i64, Shape{}, 0);           //  -> i64[](0)
    auto t4 = make_shared<Constant>(i64, Shape{}, 0);           //  -> i64[](0)
    auto t5 = make_shared<Gather>(t0, t3, t4);                  // u8[?,?,?,?], i64[], i64[] -> u8[?,?,?]
    auto t6 = make_shared<Convert>(t5, f32);                    // u8[?,?,?] -> f32[?,?,?]
    auto t7 = make_shared<Constant>(f32, Shape{1, 1, 1}, 255.0f);  //  -> f32[1,1,1]([[[255.0]]])
    auto t8 = make_shared<Divide>(t6, t7, "numpy");             // f32[?,?,?], f32[1,1,1] -> f32[?,?,?]
    auto t9 = make_shared<Constant>(i64, Shape{3}, vector<int64_t>{2, 0, 1});  //  -> i64[3]([2, 0, 1])
    auto t10 = make_shared<Transpose>(t8, t9);                  // f32[?,?,?], i64[3] -> f32[?,?,?]
    auto t11 = make_shared<Constant>(f32, Shape{3, 1, 1}, -0.5f);  //  -> f32[3,1,1]([[[-0.5]], [[-0.5]], [[-0.5]]])
    auto t12 = make_shared<Add>(t10, t11);                      // f32[?,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t13 = make_shared<Constant>(f32, Shape{3, 1, 1}, 0.5f);  //  -> f32[3,1,1]([[[0.5]], [[0.5]], [[0.5]]])
    auto t14 = make_shared<Divide>(t12, t13, "numpy");          // f32[3,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t15 = make_shared<ShapeOf>(t14);                       // f32[3,?,?] -> i64[3]
    auto t16 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t17 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t18 = make_shared<Gather>(t15, t16, t17);              // i64[3], i64[], i64[] -> i64[]
    auto t19 = make_shared<Convert>(t18, f32);                  // i64[] -> f32[]
    auto t20 = make_shared<Constant>(f32, Shape{}, 448.0);      //  -> f32[](448.0)
    auto t21 = make_shared<Divide>(t19, t20, "numpy");          // f32[], f32[] -> f32[]
    auto t22 = make_shared<Ceiling>(t21);                       // f32[] -> f32[]
    auto t23 = make_shared<Multiply>(t2, t22, "numpy");         // f32[], f32[] -> f32[]
    auto t24 = make_shared<Convert>(t23, i32);                  // f32[] -> i32[]
    auto t25 = make_shared<Convert>(t24, i64);                  // i32[] -> i64[]
    auto t26 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t27 = make_shared<Unsqueeze>(t25, t26);                // i64[], i32[] -> i64[1]
    auto t28 = make_shared<Constant>(i64, Shape{}, 2);          //  -> i64[](2)
    auto t29 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t30 = make_shared<Gather>(t15, t28, t29);              // i64[3], i64[], i64[] -> i64[]
    auto t31 = make_shared<Convert>(t30, f32);                  // i64[] -> f32[]
    auto t32 = make_shared<Divide>(t31, t20, "numpy");          // f32[], f32[] -> f32[]
    auto t33 = make_shared<Ceiling>(t32);                       // f32[] -> f32[]
    auto t34 = make_shared<Multiply>(t2, t33, "numpy");         // f32[], f32[] -> f32[]
    auto t35 = make_shared<Convert>(t34, i32);                  // f32[] -> i32[]
    auto t36 = make_shared<Convert>(t35, i64);                  // i32[] -> i64[]
    auto t37 = make_shared<Unsqueeze>(t36, t26);                // i64[], i32[] -> i64[1]
    auto t38 = make_shared<Concat>(NodeVector{t27, t37}, 0);    // i64[1], i64[1] -> i64[2]
    auto t39 = make_shared<Broadcast>(t1, t38);                 // f32[], i64[2] -> f32[?,?]
    auto t40 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t41 = make_shared<Reshape>(t39, t40, false);           // f32[?,?], i64[1] -> f32[?]
    auto t42 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t43 = make_shared<ShapeOf>(t39);                       // f32[?,?] -> i64[2]
    auto t44 = make_shared<ReduceProd>(t43, t42);               // i64[2], i64[] -> i64[]
    auto t45 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t46 = make_shared<Range>(t42, t44, t45, i64);          // i64[], i64[], i64[] -> i64[?]
    auto t47 = make_shared<Reshape>(t46, t43, false);           // i64[?], i64[2] -> i64[?,?]
    auto t48 = make_shared<Multiply>(t20, t33, "numpy");        // f32[], f32[] -> f32[]
    auto t49 = make_shared<Divide>(t48, t31, "numpy");          // f32[], f32[] -> f32[]
    auto t50 = make_shared<Multiply>(t20, t22, "numpy");        // f32[], f32[] -> f32[]
    auto t51 = make_shared<Divide>(t50, t19, "numpy");          // f32[], f32[] -> f32[]
    auto t52 = make_shared<Less>(t49, t51);                     // f32[], f32[] -> boolean[]
    auto t53 = make_shared<Convert>(t52, i64);                  // boolean[] -> i64[]
    auto t54 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t55 = make_shared<Multiply>(t53, t54, "numpy");        // i64[], i64[] -> i64[]
    auto t56 = make_shared<GreaterEqual>(t49, t51);             // f32[], f32[] -> boolean[]
    auto t57 = make_shared<Convert>(t56, i64);                  // boolean[] -> i64[]
    auto t58 = make_shared<Multiply>(t31, t51, "numpy");        // f32[], f32[] -> f32[]
    auto t59 = make_shared<Convert>(t58, i32);                  // f32[] -> i32[]
    auto t60 = make_shared<Convert>(t59, f32);                  // i32[] -> f32[]
    auto t61 = make_shared<Subtract>(t48, t60);                 // f32[], f32[] -> f32[]
    auto t62 = make_shared<Convert>(t61, i32);                  // f32[] -> i32[]
    auto t63 = make_shared<Convert>(t62, i64);                  // i32[] -> i64[]
    auto t64 = make_shared<Multiply>(t57, t63, "numpy");        // i64[], i64[] -> i64[]
    auto t65 = make_shared<Add>(t55, t64);                      // i64[], i64[] -> i64[]
    auto t66 = make_shared<Convert>(t65, i32);                  // i64[] -> i32[]
    auto t67 = make_shared<Convert>(t66, f32);                  // i32[] -> f32[]
    auto t68 = make_shared<Constant>(f32, Shape{}, 14.0);       //  -> f32[](14.0)
    auto t69 = make_shared<Divide>(t67, t68, "numpy");          // f32[], f32[] -> f32[]
    auto t70 = make_shared<Convert>(t69, i32);                  // f32[] -> i32[]
    auto t71 = make_shared<Subtract>(t35, t70);                 // i32[], i32[] -> i32[]
    auto t72 = make_shared<Convert>(t71, i64);                  // i32[] -> i64[]
    auto t73 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t74 = make_shared<Reshape>(t72, t73, false);           // i64[], i64[1] -> i64[1]
    auto t75 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t76 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t77 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t78 = make_shared<Slice>(t47, t74, t75, t76, t77);     // i64[?,?], i64[1], i64[1], i64[1], i64[1] -> i64[?,?]
    auto t79 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{-1, 1});  //  -> i64[2]([-1, 1])
    auto t80 = make_shared<Reshape>(t78, t79, false);           // i64[?,?], i64[2] -> i64[?,1]
    auto t81 = make_shared<Constant>(f32, Shape{}, 0.0);        //  -> f32[](0.0)
    auto t82 = make_shared<Slice>(t39, t74, t75, t76, t77);     // f32[?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?]
    auto t83 = make_shared<ShapeOf>(t82);                       // f32[?,?] -> i32[2]
    auto t84 = make_shared<Broadcast>(t81, t83);                // f32[], i32[2] -> f32[?,?]
    auto t85 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t86 = make_shared<Reshape>(t84, t85, false);           // f32[?,?], i64[1] -> f32[?]
    auto t87 = make_shared<ScatterNDUpdate>(t41, t80, t86);     // f32[?], i64[?,1], f32[?] -> f32[?]
    auto t88 = make_shared<Reshape>(t87, t43, false);           // f32[?], i64[2] -> f32[?,?]
    auto t89 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t90 = make_shared<Reshape>(t88, t89, false);           // f32[?,?], i64[1] -> f32[?]
    auto t91 = make_shared<Multiply>(t19, t49, "numpy");        // f32[], f32[] -> f32[]
    auto t92 = make_shared<Convert>(t91, i32);                  // f32[] -> i32[]
    auto t93 = make_shared<Convert>(t92, f32);                  // i32[] -> f32[]
    auto t94 = make_shared<Subtract>(t50, t93);                 // f32[], f32[] -> f32[]
    auto t95 = make_shared<Convert>(t94, i32);                  // f32[] -> i32[]
    auto t96 = make_shared<Convert>(t95, i64);                  // i32[] -> i64[]
    auto t97 = make_shared<Multiply>(t53, t96, "numpy");        // i64[], i64[] -> i64[]
    auto t98 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t99 = make_shared<Multiply>(t57, t98, "numpy");        // i64[], i64[] -> i64[]
    auto t100 = make_shared<Add>(t97, t99);                     // i64[], i64[] -> i64[]
    auto t101 = make_shared<Convert>(t100, i32);                // i64[] -> i32[]
    auto t102 = make_shared<Convert>(t101, f32);                // i32[] -> f32[]
    auto t103 = make_shared<Constant>(f32, Shape{}, 14.0);      //  -> f32[](14.0)
    auto t104 = make_shared<Divide>(t102, t103, "numpy");       // f32[], f32[] -> f32[]
    auto t105 = make_shared<Convert>(t104, i32);                // f32[] -> i32[]
    auto t106 = make_shared<Subtract>(t24, t105);               // i32[], i32[] -> i32[]
    auto t107 = make_shared<Convert>(t106, i64);                // i32[] -> i64[]
    auto t108 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t109 = make_shared<Reshape>(t107, t108, false);        // i64[], i64[1] -> i64[1]
    auto t110 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t111 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t112 = make_shared<Constant>(i64, Shape{1}, 0);        //  -> i64[1]([0])
    auto t113 = make_shared<Slice>(t47, t109, t110, t111, t112);  // i64[?,?], i64[1], i64[1], i64[1], i64[1] -> i64[?,?]
    auto t114 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{-1, 1});  //  -> i64[2]([-1, 1])
    auto t115 = make_shared<Reshape>(t113, t114, false);        // i64[?,?], i64[2] -> i64[?,1]
    auto t116 = make_shared<Constant>(f32, Shape{}, 0.0);       //  -> f32[](0.0)
    auto t117 = make_shared<Slice>(t88, t109, t110, t111, t112);  // f32[?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?]
    auto t118 = make_shared<ShapeOf>(t117);                     // f32[?,?] -> i32[2]
    auto t119 = make_shared<Broadcast>(t116, t118);             // f32[], i32[2] -> f32[?,?]
    auto t120 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t121 = make_shared<Reshape>(t119, t120, false);        // f32[?,?], i64[1] -> f32[?]
    auto t122 = make_shared<ScatterNDUpdate>(t90, t115, t121);  // f32[?], i64[?,1], f32[?] -> f32[?]
    auto t123 = make_shared<Reshape>(t122, t43, false);         // f32[?], i64[2] -> f32[?,?]
    auto t124 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t125 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t126 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t127 = make_shared<Gather>(t43, t125, t126);           // i64[2], i64[], i64[] -> i64[]
    auto t128 = make_shared<Constant>(i64, Shape{}, 32);        //  -> i64[](32)
    auto t129 = make_shared<Divide>(t127, t128, "numpy");       // i64[], i64[] -> i64[]
    auto t130 = make_shared<Floor>(t129);                       // i64[] -> i64[]
    auto t131 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t132 = make_shared<Unsqueeze>(t130, t131);             // i64[], i32[] -> i64[1]
    auto t133 = make_shared<Constant>(i64, Shape{1}, 32);       //  -> i64[1]([32])
    auto t134 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t135 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t136 = make_shared<Gather>(t43, t134, t135);           // i64[2], i64[], i32[] -> i64[]
    auto t137 = make_shared<Divide>(t136, t128, "numpy");       // i64[], i64[] -> i64[]
    auto t138 = make_shared<Floor>(t137);                       // i64[] -> i64[]
    auto t139 = make_shared<Unsqueeze>(t138, t131);             // i64[], i32[] -> i64[1]
    auto t140 = make_shared<Constant>(i64, Shape{1}, 32);       //  -> i64[1]([32])
    auto t141 = make_shared<Concat>(NodeVector{t124, t132, t133, t139, t140}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t142 = make_shared<Reshape>(t123, t141, false);        // f32[?,?], i64[5] -> f32[?,?,?,?,?]
    auto t143 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t144 = make_shared<Transpose>(t142, t143);             // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t145 = make_shared<Constant>(i64, Shape{3}, vector<int64_t>{-1, 32, 32});  //  -> i64[3]([-1, 32, 32])
    auto t146 = make_shared<Reshape>(t144, t145, false);        // f32[?,?,?,?,?], i64[3] -> f32[?,32,32]
    auto t147 = make_shared<Constant>(i64, Shape{2}, 0);        //  -> i64[2]([0, 0])
    auto t148 = make_shared<Constant>(i64, Shape{2}, 9223372036854775807);  //  -> i64[2]([9223372036854775807, 9223372036854775807])
    auto t149 = make_shared<Constant>(i64, Shape{2}, 2);        //  -> i64[2]([2, 2])
    auto t150 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 2});  //  -> i64[2]([1, 2])
    auto t151 = make_shared<Slice>(t146, t147, t148, t149, t150);  // f32[?,32,32], i64[2], i64[2], i64[2], i64[2] -> f32[?,16,16]
    auto t152 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t153 = make_shared<Constant>(i64, Shape{1}, 16);       //  -> i64[1]([16])
    auto t154 = make_shared<Constant>(i64, Shape{1}, 16);       //  -> i64[1]([16])
    auto t155 = make_shared<Concat>(NodeVector{t152, t132, t139, t153, t154}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t156 = make_shared<Reshape>(t151, t155, false);        // f32[?,16,16], i64[5] -> f32[?,?,?,?,?]
    auto t157 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t158 = make_shared<Transpose>(t156, t157);             // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t159 = make_shared<ShapeOf>(t158);                     // f32[?,?,?,?,?] -> i64[5]
    auto t160 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t161 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t162 = make_shared<Gather>(t159, t160, t161);          // i64[5], i64[1], i64[] -> i64[1]
    auto t163 = make_shared<Constant>(i64, Shape{1}, 2);        //  -> i64[1]([2])
    auto t164 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t165 = make_shared<Gather>(t159, t163, t164);          // i64[5], i64[1], i64[] -> i64[1]
    auto t166 = make_shared<Multiply>(t162, t165, "numpy");     // i64[1], i64[1] -> i64[1]
    auto t167 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t168 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t169 = make_shared<Gather>(t159, t167, t168);          // i64[5], i64[1], i64[] -> i64[1]
    auto t170 = make_shared<Constant>(i64, Shape{1}, 4);        //  -> i64[1]([4])
    auto t171 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t172 = make_shared<Gather>(t159, t170, t171);          // i64[5], i64[1], i64[] -> i64[1]
    auto t173 = make_shared<Multiply>(t169, t172, "numpy");     // i64[1], i64[1] -> i64[1]
    auto t174 = make_shared<Concat>(NodeVector{t166, t173}, 0);  // i64[1], i64[1] -> i64[2]
    auto t175 = make_shared<Reshape>(t158, t174, false);        // f32[?,?,?,?,?], i64[2] -> f32[?,?]
    auto t176 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{0, 1});  //  -> i32[2]([0, 1])
    auto t177 = make_shared<ReduceSum>(t175, t176);             // f32[?,?], i32[2] -> f32[]
    auto t178 = make_shared<Constant>(f32, Shape{}, 257.0);     //  -> f32[](257.0)
    auto t179 = make_shared<Add>(t177, t178);                   // f32[], f32[] -> f32[]
    auto t180 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t181 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t182 = make_shared<Gather>(t175, t180, t181);          // f32[?,?], i64[], i64[] -> f32[?]
    auto t183 = make_shared<Constant>(i32, Shape{1}, 0);        //  -> i32[1]([0])
    auto t184 = make_shared<ReduceSum>(t182, t183);             // f32[?], i32[1] -> f32[]
    auto t185 = make_shared<Add>(t179, t184);                   // f32[], f32[] -> f32[]
    auto t186 = make_shared<Constant>(f32, Shape{}, 16.0);      //  -> f32[](16.0)
    auto t187 = make_shared<Add>(t185, t186);                   // f32[], f32[] -> f32[]
    auto t188 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t189 = make_shared<Unsqueeze>(t187, t188);             // f32[], i64[] -> f32[1]
    auto t190 = make_shared<Constant>(f32, Shape{1, 32, 32}, 1.0f);  //  -> f32[1,32,32]
    auto t191 = make_shared<Concat>(NodeVector{t190, t146}, 0);  // f32[1,32,32], f32[?,32,32] -> f32[1..,32,32]
    auto t192 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t193 = make_shared<Unsqueeze>(t191, t192);             // f32[1..,32,32], i64[] -> f32[1,1..,32,32]
    auto t194 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t195 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t196 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 2});  //  -> i64[2]([1, 2])
    auto t197 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t198 = make_shared<Gather>(t15, t196, t197);           // i64[3], i64[2], i64[] -> i64[2]
    auto t199 = make_shared<Concat>(NodeVector{t194, t195, t198}, 0);  // i64[1], i64[1], i64[2] -> i64[4]
    auto t200 = make_shared<Reshape>(t14, t199, false);         // f32[3,?,?], i64[4] -> f32[?,3,?,?]
    auto t201 = make_shared<Convert>(t91, i64);                 // f32[] -> i64[]
    auto t202 = make_shared<Multiply>(t53, t201, "numpy");      // i64[], i64[] -> i64[]
    auto t203 = make_shared<Convert>(t50, i64);                 // f32[] -> i64[]
    auto t204 = make_shared<Multiply>(t57, t203, "numpy");      // i64[], i64[] -> i64[]
    auto t205 = make_shared<Add>(t202, t204);                   // i64[], i64[] -> i64[]
    auto t206 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t207 = make_shared<Unsqueeze>(t205, t206);             // i64[], i32[] -> i64[1]
    auto t208 = make_shared<Convert>(t48, i64);                 // f32[] -> i64[]
    auto t209 = make_shared<Multiply>(t53, t208, "numpy");      // i64[], i64[] -> i64[]
    auto t210 = make_shared<Convert>(t58, i64);                 // f32[] -> i64[]
    auto t211 = make_shared<Multiply>(t57, t210, "numpy");      // i64[], i64[] -> i64[]
    auto t212 = make_shared<Add>(t209, t211);                   // i64[], i64[] -> i64[]
    auto t213 = make_shared<Unsqueeze>(t212, t206);             // i64[], i32[] -> i64[1]
    auto t214 = make_shared<Concat>(NodeVector{t207, t213}, 0);  // i64[1], i64[1] -> i64[2]
    auto t215 = make_shared<Convert>(t214, i32);                // i64[2] -> i32[2]
    auto t216 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t217_attrs{Interpolate::InterpolateMode::BILINEAR_PILLOW, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t217 = make_shared<Interpolate>(t200, t215, t216, t217_attrs);  // f32[?,3,?,?], i32[2], i32[2] -> f32[?,3,?,?]
    auto t218 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t219 = make_shared<Concat>(NodeVector{t218, t207, t213}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t220 = make_shared<Reshape>(t217, t219, false);        // f32[?,3,?,?], i64[3] -> f32[?,?,?]
    auto t221 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t222 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t223 = make_shared<Convert>(t66, i64);                 // i32[] -> i64[]
    auto t224 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t225 = make_shared<Unsqueeze>(t223, t224);             // i64[], i32[] -> i64[1]
    auto t226 = make_shared<Concat>(NodeVector{t222, t207, t225}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t227 = make_shared<Broadcast>(t221, t226);             // f32[], i64[3] -> f32[3,?,?]
    auto t228 = make_shared<Concat>(NodeVector{t220, t227}, 2);  // f32[?,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t229 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t230 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t231 = make_shared<Convert>(t101, i64);                // i32[] -> i64[]
    auto t232 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t233 = make_shared<Unsqueeze>(t231, t232);             // i64[], i32[] -> i64[1]
    auto t234 = make_shared<Add>(t223, t212);                   // i64[], i64[] -> i64[]
    auto t235 = make_shared<Unsqueeze>(t234, t232);             // i64[], i32[] -> i64[1]
    auto t236 = make_shared<Concat>(NodeVector{t230, t233, t235}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t237 = make_shared<Broadcast>(t229, t236);             // f32[], i64[3] -> f32[3,?,?]
    auto t238 = make_shared<Concat>(NodeVector{t228, t237}, 1);  // f32[3,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t239 = make_shared<ShapeOf>(t238);                     // f32[3,?,?] -> i64[3]
    auto t240 = make_shared<Constant>(i64, Shape{}, 2);         //  -> i64[](2)
    auto t241 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t242 = make_shared<Gather>(t239, t240, t241);          // i64[3], i64[], i32[] -> i64[]
    auto t243 = make_shared<ShapeOf>(t228);                     // f32[3,?,?] -> i64[3]
    auto t244 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t245 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t246 = make_shared<Gather>(t243, t244, t245);          // i64[3], i64[], i64[] -> i64[]
    auto t247 = make_shared<ShapeOf>(t237);                     // f32[3,?,?] -> i64[3]
    auto t248 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t249 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t250 = make_shared<Gather>(t247, t248, t249);          // i64[3], i64[], i64[] -> i64[]
    auto t251 = make_shared<Add>(t246, t250);                   // i64[], i64[] -> i64[]
    auto t252 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t253 = make_shared<Unsqueeze>(t238, t252);             // f32[3,?,?], i64[] -> f32[1,3,?,?]
    auto t254 = make_shared<Constant>(i32, Shape{2}, 448);      //  -> i32[2]([448, 448])
    auto t255 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t256_attrs{Interpolate::InterpolateMode::CUBIC, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t256 = make_shared<Interpolate>(t253, t254, t255, t256_attrs);  // f32[1,3,?,?], i32[2], i32[2] -> f32[1,3,448,448]
    auto t257 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t258 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t259 = make_shared<Constant>(i64, Shape{}, 448);       //  -> i64[](448)
    auto t260 = make_shared<Divide>(t251, t259, "numpy");       // i64[], i64[] -> i64[]
    auto t261 = make_shared<Floor>(t260);                       // i64[] -> i64[]
    auto t262 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t263 = make_shared<Unsqueeze>(t261, t262);             // i64[], i32[] -> i64[1]
    auto t264 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t265 = make_shared<Divide>(t242, t259, "numpy");       // i64[], i64[] -> i64[]
    auto t266 = make_shared<Floor>(t265);                       // i64[] -> i64[]
    auto t267 = make_shared<Unsqueeze>(t266, t262);             // i64[], i32[] -> i64[1]
    auto t268 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t269 = make_shared<Concat>(NodeVector{t257, t258, t263, t264, t267, t268}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[6]
    auto t270 = make_shared<Reshape>(t238, t269, false);        // f32[3,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t271 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 2, 4, 1, 3, 5});  //  -> i64[6]([0, 2, 4, 1, 3, 5])
    auto t272 = make_shared<Transpose>(t270, t271);             // f32[?,?,?,?,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t273 = make_shared<Constant>(i64, Shape{4}, vector<int64_t>{-1, 3, 448, 448});  //  -> i64[4]([-1, 3, 448, 448])
    auto t274 = make_shared<Reshape>(t272, t273, false);        // f32[?,?,?,?,?,?], i64[4] -> f32[?,3,448,448]
    auto t275 = make_shared<Concat>(NodeVector{t256, t274}, 0);  // f32[1,3,448,448], f32[?,3,448,448] -> f32[1..,3,448,448]
    auto t276 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t277 = make_shared<Unsqueeze>(t275, t276);             // f32[1..,3,448,448], i64[] -> f32[1,1..,3,448,448]
    auto t278 = make_shared<Result>(t189);                      // f32[1] -> f32[1]
    t278->output(0).get_tensor().set_names({"num_img_tokens"});
    auto t279 = make_shared<Result>(t193);                      // f32[1,1..,32,32] -> f32[1,1..,32,32]
    t279->output(0).get_tensor().set_names({"image_attention_mask"});
    auto t280 = make_shared<Result>(t242);                      // i64[] -> i64[]
    t280->output(0).get_tensor().set_names({"351", "image_width"});
    auto t281 = make_shared<Result>(t251);                      // i64[] -> i64[]
    t281->output(0).get_tensor().set_names({"348", "image_height"});
    auto t282 = make_shared<Result>(t277);                      // f32[1,1..,3,448,448] -> f32[1,1..,3,448,448]
    t282->output(0).get_tensor().set_names({"input_image_embeds"});

    ResultVector results{t282, t281, t280, t279, t278};
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
    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_image_preprocessors.get()};
        ov::InferRequest& image_preprocessor = lock.get();
        image_preprocessor.set_input_tensor(image);
        image_preprocessor.infer();
        image_preprocessor.get_tensor("input_image_embeds").copy_to(input_image_embeds);
        image_preprocessor.get_tensor("image_attention_mask").copy_to(image_attention_mask);
        image_height = image_preprocessor.get_tensor("image_height").data<int64_t>()[0];
        image_width = image_preprocessor.get_tensor("image_width").data<int64_t>()[0];
        num_img_tokens = image_preprocessor.get_tensor("num_img_tokens").data<float>()[0];
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

std::pair<std::string, std::vector<size_t>> InputsEmbedderPhi4MM::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    return {phi_utils::normalize_prompt(prompt, base_id, images.size(), NATIVE_PATTERN, write_native), {}};
}

// FIXME Copied from Phi3 (except debug tensors printing and comparing) - reuse
ov::Tensor InputsEmbedderPhi4MM::get_inputs_embeds(
    const std::string& image_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence
) {
    size_t base_id = m_tokens_per_images.size();
    std::vector<ov::Tensor> images_features_proj;
    for (const ov::genai::EncodedImage& encoded_image : images) {
        images_features_proj.push_back(encoded_image.images_features_projection);
        m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
    }

    std::vector<std::variant<ov::Tensor, size_t>> new_chat_tokens;
    if (m_is_chat_conversation) {
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = phi_utils::split_tokenize(image_prompt, m_tokenizer, NATIVE_PATTERN);
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
