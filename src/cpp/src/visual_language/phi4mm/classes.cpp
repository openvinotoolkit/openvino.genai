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
    auto t9 = make_shared<Constant>(f32, Shape{3, 1, 1}, -0.5f);  //  -> f32[3,1,1]([[[-0.5]], [[-0.5]], [[-0.5]]])
    auto t10 = make_shared<Add>(t8, t9);                        // f32[?,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t11 = make_shared<Constant>(f32, Shape{3, 1, 1}, 0.5f);  //  -> f32[3,1,1]([[[0.5]], [[0.5]], [[0.5]]])
    auto t12 = make_shared<Divide>(t10, t11, "numpy");          // f32[3,?,?], f32[3,1,1] -> f32[3,?,?]
    auto t13 = make_shared<ShapeOf>(t12);                       // f32[3,?,?] -> i64[3]
    auto t14 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t15 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t16 = make_shared<Gather>(t13, t14, t15);              // i64[3], i64[], i64[] -> i64[]
    auto t17 = make_shared<Convert>(t16, f32);                  // i64[] -> f32[]
    auto t18 = make_shared<Constant>(f32, Shape{}, 448.0);      //  -> f32[](448.0)
    auto t19 = make_shared<Divide>(t17, t18, "numpy");          // f32[], f32[] -> f32[]
    auto t20 = make_shared<Ceiling>(t19);                       // f32[] -> f32[]
    auto t21 = make_shared<Multiply>(t2, t20, "numpy");         // f32[], f32[] -> f32[]
    auto t22 = make_shared<Convert>(t21, i32);                  // f32[] -> i32[]
    auto t23 = make_shared<Convert>(t22, i64);                  // i32[] -> i64[]
    auto t24 = make_shared<Constant>(i32, Shape{}, 0);          //  -> i32[](0)
    auto t25 = make_shared<Unsqueeze>(t23, t24);                // i64[], i32[] -> i64[1]
    auto t26 = make_shared<Constant>(i64, Shape{}, 2);          //  -> i64[](2)
    auto t27 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t28 = make_shared<Gather>(t13, t26, t27);              // i64[3], i64[], i64[] -> i64[]
    auto t29 = make_shared<Convert>(t28, f32);                  // i64[] -> f32[]
    auto t30 = make_shared<Divide>(t29, t18, "numpy");          // f32[], f32[] -> f32[]
    auto t31 = make_shared<Ceiling>(t30);                       // f32[] -> f32[]
    auto t32 = make_shared<Multiply>(t2, t31, "numpy");         // f32[], f32[] -> f32[]
    auto t33 = make_shared<Convert>(t32, i32);                  // f32[] -> i32[]
    auto t34 = make_shared<Convert>(t33, i64);                  // i32[] -> i64[]
    auto t35 = make_shared<Unsqueeze>(t34, t24);                // i64[], i32[] -> i64[1]
    auto t36 = make_shared<Concat>(NodeVector{t25, t35}, 0);    // i64[1], i64[1] -> i64[2]
    auto t37 = make_shared<Broadcast>(t1, t36);                 // f32[], i64[2] -> f32[?,?]
    auto t38 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t39 = make_shared<Reshape>(t37, t38, false);           // f32[?,?], i64[1] -> f32[?]
    auto t40 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t41 = make_shared<ShapeOf>(t37);                       // f32[?,?] -> i64[2]
    auto t42 = make_shared<ReduceProd>(t41, t40);               // i64[2], i64[] -> i64[]
    auto t43 = make_shared<Constant>(i64, Shape{}, 1);          //  -> i64[](1)
    auto t44 = make_shared<Range>(t40, t42, t43, i64);          // i64[], i64[], i64[] -> i64[?]
    auto t45 = make_shared<Reshape>(t44, t41, false);           // i64[?], i64[2] -> i64[?,?]
    auto t46 = make_shared<Multiply>(t18, t31, "numpy");        // f32[], f32[] -> f32[]
    auto t47 = make_shared<Divide>(t46, t29, "numpy");          // f32[], f32[] -> f32[]
    auto t48 = make_shared<Multiply>(t18, t20, "numpy");        // f32[], f32[] -> f32[]
    auto t49 = make_shared<Divide>(t48, t17, "numpy");          // f32[], f32[] -> f32[]
    auto t50 = make_shared<Less>(t47, t49);                     // f32[], f32[] -> boolean[]
    auto t51 = make_shared<Convert>(t50, i64);                  // boolean[] -> i64[]
    auto t52 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t53 = make_shared<Multiply>(t51, t52, "numpy");        // i64[], i64[] -> i64[]
    auto t54 = make_shared<GreaterEqual>(t47, t49);             // f32[], f32[] -> boolean[]
    auto t55 = make_shared<Convert>(t54, i64);                  // boolean[] -> i64[]
    auto t56 = make_shared<Multiply>(t29, t49, "numpy");        // f32[], f32[] -> f32[]
    auto t57 = make_shared<Convert>(t56, i32);                  // f32[] -> i32[]
    auto t58 = make_shared<Convert>(t57, f32);                  // i32[] -> f32[]
    auto t59 = make_shared<Subtract>(t46, t58);                 // f32[], f32[] -> f32[]
    auto t60 = make_shared<Convert>(t59, i32);                  // f32[] -> i32[]
    auto t61 = make_shared<Convert>(t60, i64);                  // i32[] -> i64[]
    auto t62 = make_shared<Multiply>(t55, t61, "numpy");        // i64[], i64[] -> i64[]
    auto t63 = make_shared<Add>(t53, t62);                      // i64[], i64[] -> i64[]
    auto t64 = make_shared<Convert>(t63, i32);                  // i64[] -> i32[]
    auto t65 = make_shared<Convert>(t64, f32);                  // i32[] -> f32[]
    auto t66 = make_shared<Constant>(f32, Shape{}, 14.0);       //  -> f32[](14.0)
    auto t67 = make_shared<Divide>(t65, t66, "numpy");          // f32[], f32[] -> f32[]
    auto t68 = make_shared<Convert>(t67, i32);                  // f32[] -> i32[]
    auto t69 = make_shared<Subtract>(t33, t68);                 // i32[], i32[] -> i32[]
    auto t70 = make_shared<Convert>(t69, i64);                  // i32[] -> i64[]
    auto t71 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t72 = make_shared<Reshape>(t70, t71, false);           // i64[], i64[1] -> i64[1]
    auto t73 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t74 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t75 = make_shared<Constant>(i64, Shape{1}, 1);         //  -> i64[1]([1])
    auto t76 = make_shared<Slice>(t45, t72, t73, t74, t75);     // i64[?,?], i64[1], i64[1], i64[1], i64[1] -> i64[?,?]
    auto t77 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{-1, 1});  //  -> i64[2]([-1, 1])
    auto t78 = make_shared<Reshape>(t76, t77, false);           // i64[?,?], i64[2] -> i64[?,1]
    auto t79 = make_shared<Constant>(f32, Shape{}, 0.0);        //  -> f32[](0.0)
    auto t80 = make_shared<Slice>(t37, t72, t73, t74, t75);     // f32[?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?]
    auto t81 = make_shared<ShapeOf>(t80);                       // f32[?,?] -> i32[2]
    auto t82 = make_shared<Broadcast>(t79, t81);                // f32[], i32[2] -> f32[?,?]
    auto t83 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t84 = make_shared<Reshape>(t82, t83, false);           // f32[?,?], i64[1] -> f32[?]
    auto t85 = make_shared<ScatterNDUpdate>(t39, t78, t84);     // f32[?], i64[?,1], f32[?] -> f32[?]
    auto t86 = make_shared<Reshape>(t85, t41, false);           // f32[?], i64[2] -> f32[?,?]
    auto t87 = make_shared<Constant>(i64, Shape{1}, -1);        //  -> i64[1]([-1])
    auto t88 = make_shared<Reshape>(t86, t87, false);           // f32[?,?], i64[1] -> f32[?]
    auto t89 = make_shared<Multiply>(t17, t47, "numpy");        // f32[], f32[] -> f32[]
    auto t90 = make_shared<Convert>(t89, i32);                  // f32[] -> i32[]
    auto t91 = make_shared<Convert>(t90, f32);                  // i32[] -> f32[]
    auto t92 = make_shared<Subtract>(t48, t91);                 // f32[], f32[] -> f32[]
    auto t93 = make_shared<Convert>(t92, i32);                  // f32[] -> i32[]
    auto t94 = make_shared<Convert>(t93, i64);                  // i32[] -> i64[]
    auto t95 = make_shared<Multiply>(t51, t94, "numpy");        // i64[], i64[] -> i64[]
    auto t96 = make_shared<Constant>(i64, Shape{}, 0);          //  -> i64[](0)
    auto t97 = make_shared<Multiply>(t55, t96, "numpy");        // i64[], i64[] -> i64[]
    auto t98 = make_shared<Add>(t95, t97);                      // i64[], i64[] -> i64[]
    auto t99 = make_shared<Convert>(t98, i32);                  // i64[] -> i32[]
    auto t100 = make_shared<Convert>(t99, f32);                 // i32[] -> f32[]
    auto t101 = make_shared<Constant>(f32, Shape{}, 14.0);      //  -> f32[](14.0)
    auto t102 = make_shared<Divide>(t100, t101, "numpy");       // f32[], f32[] -> f32[]
    auto t103 = make_shared<Convert>(t102, i32);                // f32[] -> i32[]
    auto t104 = make_shared<Subtract>(t22, t103);               // i32[], i32[] -> i32[]
    auto t105 = make_shared<Convert>(t104, i64);                // i32[] -> i64[]
    auto t106 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t107 = make_shared<Reshape>(t105, t106, false);        // i64[], i64[1] -> i64[1]
    auto t108 = make_shared<Constant>(i64, Shape{1}, 9223372036854775807);  //  -> i64[1]([9223372036854775807])
    auto t109 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t110 = make_shared<Constant>(i64, Shape{1}, 0);        //  -> i64[1]([0])
    auto t111 = make_shared<Slice>(t45, t107, t108, t109, t110);  // i64[?,?], i64[1], i64[1], i64[1], i64[1] -> i64[?,?]
    auto t112 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{-1, 1});  //  -> i64[2]([-1, 1])
    auto t113 = make_shared<Reshape>(t111, t112, false);        // i64[?,?], i64[2] -> i64[?,1]
    auto t114 = make_shared<Constant>(f32, Shape{}, 0.0);       //  -> f32[](0.0)
    auto t115 = make_shared<Slice>(t86, t107, t108, t109, t110);  // f32[?,?], i64[1], i64[1], i64[1], i64[1] -> f32[?,?]
    auto t116 = make_shared<ShapeOf>(t115);                     // f32[?,?] -> i32[2]
    auto t117 = make_shared<Broadcast>(t114, t116);             // f32[], i32[2] -> f32[?,?]
    auto t118 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t119 = make_shared<Reshape>(t117, t118, false);        // f32[?,?], i64[1] -> f32[?]
    auto t120 = make_shared<ScatterNDUpdate>(t88, t113, t119);  // f32[?], i64[?,1], f32[?] -> f32[?]
    auto t121 = make_shared<Reshape>(t120, t41, false);         // f32[?], i64[2] -> f32[?,?]
    auto t122 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t123 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t124 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t125 = make_shared<Gather>(t41, t123, t124);           // i64[2], i64[], i64[] -> i64[]
    auto t126 = make_shared<Constant>(i64, Shape{}, 32);        //  -> i64[](32)
    auto t127 = make_shared<Divide>(t125, t126, "numpy");       // i64[], i64[] -> i64[]
    auto t128 = make_shared<Floor>(t127);                       // i64[] -> i64[]
    auto t129 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t130 = make_shared<Unsqueeze>(t128, t129);             // i64[], i32[] -> i64[1]
    auto t131 = make_shared<Constant>(i64, Shape{1}, 32);       //  -> i64[1]([32])
    auto t132 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t133 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t134 = make_shared<Gather>(t41, t132, t133);           // i64[2], i64[], i32[] -> i64[]
    auto t135 = make_shared<Divide>(t134, t126, "numpy");       // i64[], i64[] -> i64[]
    auto t136 = make_shared<Floor>(t135);                       // i64[] -> i64[]
    auto t137 = make_shared<Unsqueeze>(t136, t129);             // i64[], i32[] -> i64[1]
    auto t138 = make_shared<Constant>(i64, Shape{1}, 32);       //  -> i64[1]([32])
    auto t139 = make_shared<Concat>(NodeVector{t122, t130, t131, t137, t138}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t140 = make_shared<Reshape>(t121, t139, false);        // f32[?,?], i64[5] -> f32[?,?,?,?,?]
    auto t141 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t142 = make_shared<Transpose>(t140, t141);             // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t143 = make_shared<Constant>(i64, Shape{3}, vector<int64_t>{-1, 32, 32});  //  -> i64[3]([-1, 32, 32])
    auto t144 = make_shared<Reshape>(t142, t143, false);        // f32[?,?,?,?,?], i64[3] -> f32[?,32,32]
    auto t145 = make_shared<Constant>(i64, Shape{2}, 0);        //  -> i64[2]([0, 0])
    auto t146 = make_shared<Constant>(i64, Shape{2}, 9223372036854775807);  //  -> i64[2]([9223372036854775807, 9223372036854775807])
    auto t147 = make_shared<Constant>(i64, Shape{2}, 2);        //  -> i64[2]([2, 2])
    auto t148 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 2});  //  -> i64[2]([1, 2])
    auto t149 = make_shared<Slice>(t144, t145, t146, t147, t148);  // f32[?,32,32], i64[2], i64[2], i64[2], i64[2] -> f32[?,16,16]
    auto t150 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t151 = make_shared<Constant>(i64, Shape{1}, 16);       //  -> i64[1]([16])
    auto t152 = make_shared<Constant>(i64, Shape{1}, 16);       //  -> i64[1]([16])
    auto t153 = make_shared<Concat>(NodeVector{t150, t130, t137, t151, t152}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[5]
    auto t154 = make_shared<Reshape>(t149, t153, false);        // f32[?,16,16], i64[5] -> f32[?,?,?,?,?]
    auto t155 = make_shared<Constant>(i64, Shape{5}, vector<int64_t>{0, 1, 3, 2, 4});  //  -> i64[5]([0, 1, 3, 2, 4])
    auto t156 = make_shared<Transpose>(t154, t155);             // f32[?,?,?,?,?], i64[5] -> f32[?,?,?,?,?]
    auto t157 = make_shared<ShapeOf>(t156);                     // f32[?,?,?,?,?] -> i64[5]
    auto t158 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t159 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t160 = make_shared<Gather>(t157, t158, t159);          // i64[5], i64[1], i64[] -> i64[1]
    auto t161 = make_shared<Constant>(i64, Shape{1}, 2);        //  -> i64[1]([2])
    auto t162 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t163 = make_shared<Gather>(t157, t161, t162);          // i64[5], i64[1], i64[] -> i64[1]
    auto t164 = make_shared<Multiply>(t160, t163, "numpy");     // i64[1], i64[1] -> i64[1]
    auto t165 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t166 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t167 = make_shared<Gather>(t157, t165, t166);          // i64[5], i64[1], i64[] -> i64[1]
    auto t168 = make_shared<Constant>(i64, Shape{1}, 4);        //  -> i64[1]([4])
    auto t169 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t170 = make_shared<Gather>(t157, t168, t169);          // i64[5], i64[1], i64[] -> i64[1]
    auto t171 = make_shared<Multiply>(t167, t170, "numpy");     // i64[1], i64[1] -> i64[1]
    auto t172 = make_shared<Concat>(NodeVector{t164, t171}, 0);  // i64[1], i64[1] -> i64[2]
    auto t173 = make_shared<Reshape>(t156, t172, false);        // f32[?,?,?,?,?], i64[2] -> f32[?,?]
    auto t174 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{0, 1});  //  -> i32[2]([0, 1])
    auto t175 = make_shared<ReduceSum>(t173, t174);             // f32[?,?], i32[2] -> f32[]
    auto t176 = make_shared<Constant>(f32, Shape{}, 257.0);     //  -> f32[](257.0)
    auto t177 = make_shared<Add>(t175, t176);                   // f32[], f32[] -> f32[]
    auto t178 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t179 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t180 = make_shared<Gather>(t173, t178, t179);          // f32[?,?], i64[], i64[] -> f32[?]
    auto t181 = make_shared<Constant>(i32, Shape{1}, 0);        //  -> i32[1]([0])
    auto t182 = make_shared<ReduceSum>(t180, t181);             // f32[?], i32[1] -> f32[]
    auto t183 = make_shared<Add>(t177, t182);                   // f32[], f32[] -> f32[]
    auto t184 = make_shared<Constant>(f32, Shape{}, 16.0);      //  -> f32[](16.0)
    auto t185 = make_shared<Add>(t183, t184);                   // f32[], f32[] -> f32[]
    auto t186 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t187 = make_shared<Unsqueeze>(t185, t186);             // f32[], i64[] -> f32[1]
    auto t188 = make_shared<Constant>(f32, Shape{1, 32, 32}, 1.0f);  //  -> f32[1,32,32]
    auto t189 = make_shared<Concat>(NodeVector{t188, t144}, 0);  // f32[1,32,32], f32[?,32,32] -> f32[1..,32,32]
    auto t190 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t191 = make_shared<Unsqueeze>(t189, t190);             // f32[1..,32,32], i64[] -> f32[1,1..,32,32]
    auto t192 = make_shared<Constant>(i64, Shape{1}, -1);       //  -> i64[1]([-1])
    auto t193 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t194 = make_shared<Constant>(i64, Shape{2}, vector<int64_t>{1, 2});  //  -> i64[2]([1, 2])
    auto t195 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t196 = make_shared<Gather>(t13, t194, t195);           // i64[3], i64[2], i64[] -> i64[2]
    auto t197 = make_shared<Concat>(NodeVector{t192, t193, t196}, 0);  // i64[1], i64[1], i64[2] -> i64[4]
    auto t198 = make_shared<Reshape>(t12, t197, false);         // f32[3,?,?], i64[4] -> f32[?,3,?,?]
    auto t199 = make_shared<Convert>(t89, i64);                 // f32[] -> i64[]
    auto t200 = make_shared<Multiply>(t51, t199, "numpy");      // i64[], i64[] -> i64[]
    auto t201 = make_shared<Convert>(t48, i64);                 // f32[] -> i64[]
    auto t202 = make_shared<Multiply>(t55, t201, "numpy");      // i64[], i64[] -> i64[]
    auto t203 = make_shared<Add>(t200, t202);                   // i64[], i64[] -> i64[]
    auto t204 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t205 = make_shared<Unsqueeze>(t203, t204);             // i64[], i32[] -> i64[1]
    auto t206 = make_shared<Convert>(t46, i64);                 // f32[] -> i64[]
    auto t207 = make_shared<Multiply>(t51, t206, "numpy");      // i64[], i64[] -> i64[]
    auto t208 = make_shared<Convert>(t56, i64);                 // f32[] -> i64[]
    auto t209 = make_shared<Multiply>(t55, t208, "numpy");      // i64[], i64[] -> i64[]
    auto t210 = make_shared<Add>(t207, t209);                   // i64[], i64[] -> i64[]
    auto t211 = make_shared<Unsqueeze>(t210, t204);             // i64[], i32[] -> i64[1]
    auto t212 = make_shared<Concat>(NodeVector{t205, t211}, 0);  // i64[1], i64[1] -> i64[2]
    auto t213 = make_shared<Convert>(t212, i32);                // i64[2] -> i32[2]
    auto t214 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t215_attrs{Interpolate::InterpolateMode::BILINEAR_PILLOW, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t215 = make_shared<Interpolate>(t198, t213, t214, t215_attrs);  // f32[?,3,?,?], i32[2], i32[2] -> f32[?,3,?,?]
    auto t216 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t217 = make_shared<Concat>(NodeVector{t216, t205, t211}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t218 = make_shared<Reshape>(t215, t217, false);        // f32[?,3,?,?], i64[3] -> f32[?,?,?]
    auto t219 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t220 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t221 = make_shared<Convert>(t64, i64);                 // i32[] -> i64[]
    auto t222 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t223 = make_shared<Unsqueeze>(t221, t222);             // i64[], i32[] -> i64[1]
    auto t224 = make_shared<Concat>(NodeVector{t220, t205, t223}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t225 = make_shared<Broadcast>(t219, t224);             // f32[], i64[3] -> f32[3,?,?]
    auto t226 = make_shared<Concat>(NodeVector{t218, t225}, 2);  // f32[?,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t227 = make_shared<Constant>(f32, Shape{}, 1.0);       //  -> f32[](1.0)
    auto t228 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t229 = make_shared<Convert>(t99, i64);                 // i32[] -> i64[]
    auto t230 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t231 = make_shared<Unsqueeze>(t229, t230);             // i64[], i32[] -> i64[1]
    auto t232 = make_shared<Add>(t221, t210);                   // i64[], i64[] -> i64[]
    auto t233 = make_shared<Unsqueeze>(t232, t230);             // i64[], i32[] -> i64[1]
    auto t234 = make_shared<Concat>(NodeVector{t228, t231, t233}, 0);  // i64[1], i64[1], i64[1] -> i64[3]
    auto t235 = make_shared<Broadcast>(t227, t234);             // f32[], i64[3] -> f32[3,?,?]
    auto t236 = make_shared<Concat>(NodeVector{t226, t235}, 1);  // f32[3,?,?], f32[3,?,?] -> f32[3,?,?]
    auto t237 = make_shared<ShapeOf>(t236);                     // f32[3,?,?] -> i64[3]
    auto t238 = make_shared<Constant>(i64, Shape{}, 2);         //  -> i64[](2)
    auto t239 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t240 = make_shared<Gather>(t237, t238, t239);          // i64[3], i64[], i32[] -> i64[]
    auto t241 = make_shared<ShapeOf>(t226);                     // f32[3,?,?] -> i64[3]
    auto t242 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t243 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t244 = make_shared<Gather>(t241, t242, t243);          // i64[3], i64[], i64[] -> i64[]
    auto t245 = make_shared<ShapeOf>(t235);                     // f32[3,?,?] -> i64[3]
    auto t246 = make_shared<Constant>(i64, Shape{}, 1);         //  -> i64[](1)
    auto t247 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t248 = make_shared<Gather>(t245, t246, t247);          // i64[3], i64[], i64[] -> i64[]
    auto t249 = make_shared<Add>(t244, t248);                   // i64[], i64[] -> i64[]
    auto t250 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t251 = make_shared<Unsqueeze>(t236, t250);             // f32[3,?,?], i64[] -> f32[1,3,?,?]
    auto t252 = make_shared<Constant>(i32, Shape{2}, 448);      //  -> i32[2]([448, 448])
    auto t253 = make_shared<Constant>(i32, Shape{2}, vector<int32_t>{2, 3});  //  -> i32[2]([2, 3])
    Interpolate::InterpolateAttrs t254_attrs{Interpolate::InterpolateMode::CUBIC, Interpolate::ShapeCalcMode::SIZES, vector<size_t>{0, 0, 0, 0}, vector<size_t>{0, 0, 0, 0}, Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL, Interpolate::NearestMode::FLOOR, false, -0.75};
    auto t254 = make_shared<Interpolate>(t251, t252, t253, t254_attrs);  // f32[1,3,?,?], i32[2], i32[2] -> f32[1,3,448,448]
    auto t255 = make_shared<Constant>(i64, Shape{1}, 1);        //  -> i64[1]([1])
    auto t256 = make_shared<Constant>(i64, Shape{1}, 3);        //  -> i64[1]([3])
    auto t257 = make_shared<Constant>(i64, Shape{}, 448);       //  -> i64[](448)
    auto t258 = make_shared<Divide>(t249, t257, "numpy");       // i64[], i64[] -> i64[]
    auto t259 = make_shared<Floor>(t258);                       // i64[] -> i64[]
    auto t260 = make_shared<Constant>(i32, Shape{}, 0);         //  -> i32[](0)
    auto t261 = make_shared<Unsqueeze>(t259, t260);             // i64[], i32[] -> i64[1]
    auto t262 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t263 = make_shared<Divide>(t240, t257, "numpy");       // i64[], i64[] -> i64[]
    auto t264 = make_shared<Floor>(t263);                       // i64[] -> i64[]
    auto t265 = make_shared<Unsqueeze>(t264, t260);             // i64[], i32[] -> i64[1]
    auto t266 = make_shared<Constant>(i64, Shape{1}, 448);      //  -> i64[1]([448])
    auto t267 = make_shared<Concat>(NodeVector{t255, t256, t261, t262, t265, t266}, 0);  // i64[1], i64[1], i64[1], i64[1], i64[1], i64[1] -> i64[6]
    auto t268 = make_shared<Reshape>(t236, t267, false);        // f32[3,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t269 = make_shared<Constant>(i64, Shape{6}, vector<int64_t>{0, 2, 4, 1, 3, 5});  //  -> i64[6]([0, 2, 4, 1, 3, 5])
    auto t270 = make_shared<Transpose>(t268, t269);             // f32[?,?,?,?,?,?], i64[6] -> f32[?,?,?,?,?,?]
    auto t271 = make_shared<Constant>(i64, Shape{4}, vector<int64_t>{-1, 3, 448, 448});  //  -> i64[4]([-1, 3, 448, 448])
    auto t272 = make_shared<Reshape>(t270, t271, false);        // f32[?,?,?,?,?,?], i64[4] -> f32[?,3,448,448]
    auto t273 = make_shared<Concat>(NodeVector{t254, t272}, 0);  // f32[1,3,448,448], f32[?,3,448,448] -> f32[1..,3,448,448]
    auto t274 = make_shared<Constant>(i64, Shape{}, 0);         //  -> i64[](0)
    auto t275 = make_shared<Unsqueeze>(t273, t274);             // f32[1..,3,448,448], i64[] -> f32[1,1..,3,448,448]
    auto t276 = make_shared<Result>(t187);                      // f32[1] -> f32[1]
    t276->output(0).get_tensor().set_names({"num_img_tokens"});
    auto t277 = make_shared<Result>(t191);                      // f32[1,1..,32,32] -> f32[1,1..,32,32]
    t277->output(0).get_tensor().set_names({"image_attention_mask"});
    auto t278 = make_shared<Result>(t240);                      // i64[] -> i64[]
    t278->output(0).get_tensor().set_names({"344", "image_width"});
    auto t279 = make_shared<Result>(t249);                      // i64[] -> i64[]
    t279->output(0).get_tensor().set_names({"341", "image_height"});
    auto t280 = make_shared<Result>(t275);                      // f32[1,1..,3,448,448] -> f32[1,1..,3,448,448]
    t280->output(0).get_tensor().set_names({"input_image_embeds"});

    ResultVector results{t280, t279, t278, t277, t276};
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

EncodedImage VisionEncoderPhi4MM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);
    ov::Tensor input_image_embeds, image_attention_mask, patch_position_ids;
    int32_t image_height = 0, image_width = 0, num_img_tokens = 0;
    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_image_preprocessors.get()};
        ov::InferRequest& image_preprocessor = lock.get();
        image_preprocessor.set_input_tensor(image);
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
    EncodedImage encoded_image;
    {
        CircularBufferQueueElementGuard<ov::InferRequest> lock{m_ireq_queue_vision_projection.get()};
        ov::InferRequest& projector = lock.get();
        projector.set_input_tensor(_1le);
        projector.infer();
        projector.get_output_tensor().copy_to(encoded_image.resized_source);
    }
    std::cout << "DDDDDDDDDDDDDDDDd\n";
    // assert projected.back().get_shape().at(1) == tokens_per_images
    return encoded_image;

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
    // EncodedImage encoded_image;

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

std::pair<std::string, std::vector<size_t>> InputsEmbedderPhi4MM::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    return {util::normalize_prompt(prompt, base_id, images.size(), NATIVE_PATTERN, write_native), {}};
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
        new_chat_tokens = util::split_tokenize(image_prompt, m_tokenizer, NATIVE_PATTERN);
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
        new_chat_tokens = util::split_tokenize(templated_prompt, m_tokenizer, NATIVE_PATTERN);
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

} // namespace ov::genai
