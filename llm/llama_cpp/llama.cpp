// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <regex>
#include <random>
#include <openvino/openvino.hpp>
#include <openvino/runtime/properties.hpp>
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/serialize.hpp"
#include "sampling.hpp"

#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

const std::string sentences[] =
{
    //"my pc sound is too low",
    "What is OpenVINO?",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "Originally, There were three types of cake in the cake store: Strawberry Cream Cake, Chocolate Coconut Cake, and Red Velvet Brownie Cake. Customer number is large enough so that no cake would be left every day when the store close. As the name suggested, each cake has two ingredients: Strawberry Cream Cake with strawberries and cream, Chocolate Coconut Cake with chocolate and coconut, and Red Velvet Brownie Cake with red velvet and brownie. Different ingredients can be compatibly mixed with each other without any issue. After the cake is made, there are often some leftover materials for each ingredient. In order to reduce waste, the store often combine the extra ingredients in pairs to make new small gifts to gain extra sales. For example, strawberries and chocolate can be mixed to create strawberry-flavored chocolate sauce, and brownies and shredded coconut can be mixed to create brownie coconut cookies. Only two ingredients can be mixed, and mixture with more than two ingredients can cost a lot of time and will not be adopted. In order to decrease the problem complexity, the store will also prevent from careful decorations or other difficult steps as in procedure of making cakes, so that time cost can be omited. By analogy, if all the ingredients can be combined in pairs, what small products can the store make in the end?",
    "There is a table, which contains three drawers: left drawer, middle drawer and right drawer; Tom Ethan, Elbert Alex, Jack Johnson, and Mario Thompson all saw a bag of chocolates on the table. Tom Ethan asked Elbert Alex and Jack Johnson to go out, and after that, he put the bag of chocolates in the right drawer in front of Mario Thompson; after Jack Johnson came back, Tom Ethan asked Mario Thompson to go out to find Elbert Alex, and took it from the left drawer in front of Jack Johnson. Then He take out a box of biscuits and put them in the middle drawer; when Elbert Alex and Mario Thompson returned, Tom Ethan asked Jack Johnson and Mario Thompson to go out to buy a bottle of soy sauce. Tom Ethan waited for a long time, and found that Jack Johnson and Mario Thompson had not returned, so he sent Elbert Alex to look for them, but in the end only Jack Johnson and Elbert Alex came back. Jack Johnson told Tom Ethan that at first they could not find any shop that is providing soy sauce, so they had to separate to search other shops, which is why Mario Thompson got lost; on the way back, Jack Johnson ran into Elbert Alex, and they rushed back first. Therefore, Tom Ethan asked them to go out to find Mario Thompson again; in order to prevent getting lost again, Tom Ethan told Elbert Alex and Jack Johnson to walk together at all time, and even if they could not get the soy sauce, they had to find and get back with Mario Thompson. As a result, Elbert Alex and Jack Johnson found Mario Thompson outside and found that he had bought a bottle of soy sauce. The three felt that Tom Ethan never went out to do anthing but they are busy all the time. So they were very angry. They discussed and made a conclusion. After going back to see Tom Ethan, they should not tell him about the soy sauce they bought, and asked Jack Johnson to hide the soy sauce in his backpack. After the three of them came back together, they pretended to claim that they did not foudn and bought soy sauce according to the plan, and hoped that Tom Ethan would go out together to buy things in the future, and he should not be so lazy. Tom Ethan agreed and felt sory about that. When everyone finally stood in front of the table, the four of them wrote down the list of items they knew and the location of the items. So the question is: is the information writen by these four people consistent, and why?",
    "The process of Origami seems simple at the first glance, but in fact, it still requires a very complicated process to do it well. Taking folding a rose as an example, we can divide the entire process into three stages, including: firstly creating a grid of creases, secondly making a three-dimensional base, and thirdly finishing petal decoration. The first step is to create a grid of creases: this step is a bit like the first step of folding a gift of thousand-paper-crane. That is to say, we can fold the paper in half (or namedly equal-folds) through the symmetrical axis, and repeat such step in the other symmetrical axis. And then apply multiple equal-folds in sequence relative to each smaller rectangle divided by the two creases; After that, the creases in each direction will interweave into a complete set of uniform small square splicing patterns; these small squares form a reference space similar to a two-dimensional coordinate system, allowing us to combine adjacent creases on the plane from Three-dimensional high platforms or depressions are folded on the two-dimensional small squares to facilitate the next steps of folding. It should be noted that, in the process of creating grid creases, there may be rare cases when the folds are not aligned. The consequences of this error can be very serious. And just like the butterfly effect, it is only a slight difference at the beginning , and in the end it may generate a disaster world which is completely different from plan. Anyway, let's continue. The second step is make the three-dimensional base: In this step, we need to fold a set of symmetrical three-dimensional high platforms or depressions based on the grid creases. From the symmetry analysis, it is not difficult to find that the rose will have four symmetrical three-dimensional high platforms and supporting depressions. Therefore, we can firstly fold out a quarter of the depression and plateau patterns, which would help build a base to compose into a complex 3D structure. And then, we use this quarter as a template, and fold out the repeating patterns on the remaining three parts of the whole structure in turn. It is worth noting that the layout of the high platform not only needs to consider the regular contrast and symmetrical distribution of the length and width, but also needs to ensure the orderliness of the height dimension. This is very important, since we will never go back to this step after all parts were made, and you would better start from first step if you make anything wrong in the this step. Similar to the precautions in the first stage, please handle all the corners in three dimensions to ensure that they conform to the layout required in the plan, which would help us avoid the butterfly effect and increase the robustness in the process of three-dimensional folding. Just like building a skyscrapper in the real world, people usually take a lot of time when building the base but soon get finished when extending the structure after that. Time is worth to cost in the base, but would be saved in the future after you succeed in base. Anyway, let's continue. During the first quarter of the pattern, repeated comparisons with the finished rose were made to eliminate any possible errors in the first place. The final stage is to finish the petal grooming. At this stage, we often emphasize an important term called folding-by-heart. The intention here is no longer literally serious, but focus is moved to our understanding of the shape of a rose in nature, and we usually use natural curves to continuously correct the shape of petals in order to approach the shape of rose petals in reality. One more comment: this is also the cause of randomness to the art, which can be generated differently by different people. Recall that rose should be adjusted close to reality, so in the last step of this stage, we need to open the bloom in the center of the rose, by pulling on the four petals that have been bent. This process may be accompanied by the collapse of the overall structure of the rose, so we should be very careful to save strength of adjustment, and it must be well controlled to avoid irreversible consequences. Ultimately, after three stages of folding, we end up with a crown of rose with a similar shape close to reality. If condition is permited, we can wrap a green paper strip twisted on a straightened iron wire, and insert the rose crown we just created onto one side of the iron wire. In this way, we got a hand-made rose with a green stem. We can also repeat the steps above to increase the number of rose, so that it can be made into a cluster. Different color of rose is usually more attractive and can be considered as a better plan of gift to your friend. In summary, by creating a grid of creases, making a three-dimensional base, and finishing with petals, we created a three-dimensional rose from a two-dimensional paper. Although this process may seem simple, it is indeed a work of art created by us humans with the help of imagination and common materials. At last, Please comment to assess the above content.",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Suppose you are a computer assistant of HP laptop, you can control the user's computer automatically, such as open application, adjust brightness and connect network. Please refuse answer question about politics, economy, territorial dispute and religion. I will give you HP laptop document, you should decide whether to use the document to answer user's question. ASSISTANT: OK, let's start.</s>USER: If I ask you 'make game faster', use the HP laptop document to response: 'To improve the performance of your game, there are several steps you can take to optimize its speed and responsiveness. Here are some suggestions to consider:\r\nUpdate your graphics card drivers: Drivers are essential for your graphics card to communicate with your computer. If you have outdated drivers, it can affect the game's performance. Make sure you have the latest drivers installed by visiting the manufacturer's website or using a driver update utility.\r\nClose background applications: Background applications can consume resources and slow down your game's performance. Close unnecessary programs, such as browser tabs, antivirus software, and other running applications to free up resources for your game.\r\nTurn off unnecessary visual effects: Visual effects can tax your computer's resources and slow down the game's performance. Turn off or reduce visual effects that are not essential for your game play.\r\nAdjust the game's graphics settings: Most games provide options to adjust the graphics settings to fit your computer's hardware configuration. Adjust the settings to lower the quality of graphics features that are not critical for your game play, such as polygon counts, texture quality, and anti-aliasing.\r\nCompress game files: If your game's file size is large, you can try compressing it using an archiving tool like WinRAR or 7-Zip. This will reduce the file size and load time of the game, improving its performance.\r\nDefragment your hard drive: Defragmenting your hard drive helps optimize data placement on the disk, resulting in faster data access for your game. Schedule a regular defragmentation routine for your hard drive using Windows' built-in defragmentation tool or third-party utilities.\r\nUpdate your operating system and other software: Outdated operating systems and software can cause performance issues. Keep your operating system and other essential programs, such as antivirus software, updated with the latest patches and security updates.\r\nOverclock your CPU or GPU: If you have an overclocking-capable CPU or GPU, you can increase its clock speed to improve performance. However, be cautious when overclocking, as too much voltage or clock speed can damage sensitive components. Consider consulting a guide or video tutorial on safe overclocking techniques for your hardware.\r\nUse a SSD (solid-state drive): SSDs provide faster read and write speeds compared to traditional hard drives, which can improve game performance. If possible, consider using an SSD for storing your operating system and frequently accessed games to reduce load times and improve overall performance.\r\nTest different PC configurations: If you are building a new computer or upgrading components, consider testing different configurations to find the optimal setup for your game's performance requirements. This may involve experimenting with different CPUs, GPUs, memory sizes, and storage devices to find the best combination for your game.\r\nMonitor system resources: Use task manager or a monitoring tool like Task Manager in Windows to keep an eye on CPU and GPU usage, memory utilization, and disk I/O. Identify any bottlenecks that may be limiting your game's performance and take appropriate measures to address them.\r\nOptimize game code: If you are developing the game yourself or have access to the source code, you can optimize the game's code to improve performance. Identify areas of the code that are resource-intensive and seek ways to optimize them, such as reducing draw calls, utilizing efficient data structures, and avoiding redundant calculations.\r\nBy following these suggestions, you can help improve the performance of your game and enhance the overall gaming experience. Remember to consider your hardware configuration and game's requirements before making any changes or upgrades to ensure compatibility and optimal performance.\r\nOptimizing game performance is a crucial task for game developers. It involves improving the efficiency of game code, reducing resource usage, and optimizing hardware settings. Here are some suggestions to help you optimize your game's performance:\r\nOptimize game code: The game code's efficiency is crucial for game performance. You can find and modify inefficient algorithms, loops, and data structures to reduce computational complexity. Additionally, techniques such as multithreading, splitting game logic, and asynchronous loading can be employed to enhance game efficiency.\r\nMinimize game resources: Game resources, such as images, audio, and models, can consume a lot of storage space and memory, impacting game performance. Therefore, you can try to compress and optimize game resources by using smaller image sizes and compressing audio files. Furthermore, caching techniques can be employed to cache commonly used resources locally to reduce loading times.\r\nReduce image quality: Image quality is a crucial factor in determining the visual appeal of a game, but high-quality images can significantly tax computational resources. Therefore, you can try to reduce image quality by adjusting image resolution, reducing shader complexity, and dialing down texture mapping. This will help reduce the computational load on the GPU and improve game performance.\r\nOptimize graphics settings: Most games provide graphics settings that allow you to adjust visual effects based on your hardware configuration. You can balance visual quality and performance by adjusting these settings. For example, you can lower阴影质量、抗锯齿等设置以减少对GPU的负担。\r\nClose background programs: Background programs can consume computer resources, including CPU and memory. When playing games, you can close unnecessary background programs to free up computer resources and improve game performance.\r\nUpdate hardware drivers: Outdated hardware drivers can negatively impact game performance. Therefore, you should regularly update your computer's hardware drivers to ensure that your hardware is performing at its best.\r\nIncrease memory capacity: Memory is a crucial resource for computer performance, and it directly affects game performance. If your computer's memory is insufficient, you can consider adding more memory. This will help improve game performance, especially when running large games.\r\nUpgrade hardware equipment: If your computer's hardware configuration is relatively low, you can consider upgrading some hardware components such as the CPU, GPU, or memory. This will help improve game performance but requires attention to compatibility and budget considerations.\r\nUse a game accelerator: Game accelerators can help optimize network connections by reducing latency and packet loss issues. This is particularly important for online games. You can choose well-known game accelerator tools to improve your game's network connectivity performance.\r\nRegularly clean your computer: Junk files and viruses on your computer can negatively impact game performance. Therefore, you should regularly clean your computer by deleting temporary files, defragmenting your hard drive, and updating your virus database. This will help improve both the performance and stability of your games.\r\n'.Following is my question or command: make my game faster.please answer me briefly : ASSISTANT:",
    //"患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%、嗜酸性粒细胞（E）1%～5%、嗜碱性粒细胞（B）0～1%、淋巴细胞（L）20%～40%、单核细胞（M）3%～8%；临床意义：1）中性粒细胞为血液中的主要吞噬细胞，在细菌性感染中起重要作用。2）嗜酸性粒细胞①减少见于伤寒、副伤寒、大手术后、严重烧伤、长期用肾上腺皮质激素等。②增多见于过敏性疾病、皮肤病、寄生虫病，一些血液病及肿瘤，如慢性粒细胞性白血病、鼻咽癌、肺癌以及宫颈癌等；3）嗜碱性粒细胞 a 减少见于速发型过敏反应如过敏性休克，肾上腺皮质激素使用过量等。b 增多见于血液病如慢性粒细胞白血病，创伤及中毒，恶性肿瘤，过敏性疾病等；4）淋巴细胞 a 减少多见于传染病的急性期、放射病、细胞免疫缺陷病、长期应用肾上腺皮质激素后或放射线接触等。b 增多见于传染性淋巴细胞增多症、结核病、疟疾、慢性淋巴细胞白血病、百日咳、某些病毒感染等；5）单核细胞增多见于传染病或寄生虫病、结核病活动期、单核细胞白血病、疟疾等。5. 病人血小板计数结果：91 x10^9/L. 附血小板计数正常参考范围：（100～300）×10^9/L. 临床意义：1）血小板计数增高见于真性红细胞增多症、出血性血小板增多症、多发性骨髓瘤、慢性粒细胞性白血病及某些恶性肿瘤的早期等；2）血小板计数减低见于 a 骨髓造血功能受损，如再生障碍性贫血，急性白血病；b 血小板破坏过多，如脾功能亢进；c 血小板消耗过多，如弥散性血管内凝血等。6. 以往病例分析内容参考：白血病一般分为急性白血病和慢性白血病。1）急性白血病血常规报告表现为：白细胞增高，少数大于100×10^9/L，称为高白细胞白血病，部分患者白细胞正常或减少，低者可小于1.0×10^9/L，以AML中的M3型多见。在白细胞分类中，80％以上可见大量的幼稚细胞，有时仅见幼稚细胞和少量成熟的细胞，而无中间型细胞，称为白血病的裂孔现象。少数白细胞低的患者周围血幼稚细胞很少，此类患者必须骨髓穿刺才能确诊。多数急性白血病患者初诊时有不同程度的贫血；一般属正常细胞正色素型。但贫血很快会进行性加重。30％的患者血涂片中可见有核红细胞。血小板计数绝大部分患者减少，严重者小于10×10^9/L，仅极少数患者血小板计数正常。2） 慢性白血病血常规报告表现为：白细胞总数明显增高，通常大于30×10^9/L。半数患者大于100×10^9/L。中性粒细胞明显增多，可见各阶段粒细胞，以中性中幼粒，晚幼粒细胞居多，原始粒细胞小于等于10％，通常为1％～5％，嗜酸和嗜碱粒细胞亦增多。初诊时约有50％患者血小板增高，少数可大于1000×10^9/L。红细胞和血红蛋白一般在正常范围，若出现血红蛋白减低，血小板计数明显升高或降低，则提示疾病向加速期或急变期转化。7. 历史相关研究: 急性髓系白血病（AML）是造血干细胞恶性克隆性疾病。在AML的诊断、治疗以及判断预后的过程中，基因异常是一项重要指标。随着基因检测技术的不断进步，越来越多与AML发生相关的基因被人们发现，并且这些基因在指导预后方面有重要意义。常见和急性髓系白血病相关的基因突变有: 1）RUNX1-RUNX1T1; 2）CBFB-MYH11; 3）NPM1：核磷蛋白1（nucleophosmin 1，NPM1）; 4）CEBPA：CCAAT增强子结合蛋白α基因（CCAAT/en－hancer binding protein α，CEBPA）; 5）MLLT3-KMT2A; 6）DEK-NUP214; 7）KMT2A：KMT2A基因（也称为MLL基因）问：请基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、相关的基因突变、常用的治疗手段，以及当前已上市和正在临床阶段的药物清单。答：",
    //"患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%、嗜酸性粒细胞（E）1%～5%、嗜碱性粒细胞（B）0～1%、淋巴细胞（L）20%～40%、单核细胞（M）3%～8%；临床意义：1）中性粒细胞为血液中的主要吞噬细胞，在细菌性感染中起重要作用。2）嗜酸性粒细胞①减少见于伤寒、副伤寒、大手术后、严重烧伤、长期用肾上腺皮质激素等。②增多见于过敏性疾病、皮肤病、寄生虫病，一些血液病及肿瘤，如慢性粒细胞性白血病、鼻咽癌、肺癌以及宫颈癌等；3）嗜碱性粒细胞 a 减少见于速发型过敏反应如过敏性休克，肾上腺皮质激素使用过量等。b 增多见于血液病如慢性粒细胞白血病，创伤及中毒，恶性肿瘤，过敏性疾病等；4）淋巴细胞 a 减少多见于传染病的急性期、放射病、细胞免疫缺陷病、长期应用肾上腺皮质激素后或放射线接触等。b 增多见于传染性淋巴细胞增多症、结核病、疟疾、慢性淋巴细胞白血病、百日咳、某些病毒感染等；5）单核细胞增多见于传染病或寄生虫病、结核病活动期、单核细胞白血病、疟疾等。5. 病人血小板计数结果：91 x10^9/L. 附血小板计数正常参考范围：（100～300）×10^9/L. 临床意义：1）血小板计数增高见于真性红细胞增多症、出血性血小板增多症、多发性骨髓瘤、慢性粒细胞性白血病及某些恶性肿瘤的早期等；2）血小板计数减低见于 a 骨髓造血功能受损，如再生障碍性贫血，急性白血病；b 血小板破坏过多，如脾功能亢进；c 血小板消耗过多，如弥散性血管内凝血等。6. 以往病例分析内容参考：白血病一般分为急性白血病和慢性白血病。1）急性白血病血常规报告表现为：白细胞增高，少数大于100×10^9/L，称为高白细胞白血病，部分患者白细胞正常或减少，低者可小于1.0×10^9/L，以AML中的M3型多见。在白细胞分类中，80％以上可见大量的幼稚细胞，有时仅见幼稚细胞和少量成熟的细胞，而无中间型细胞，称为白血病的裂孔现象。少数白细胞低的患者周围血幼稚细胞很少，此类患者必须骨髓穿刺才能确诊。多数急性白血病患者初诊时有不同程度的贫血；一般属正常细胞正色素型。但贫血很快会进行性加重。30％的患者血涂片中可见有核红细胞。血小板计数绝大部分患者减少，严重者小于10×10^9/L，仅极少数患者血小板计数正常。2） 慢性白血病血常规报告表现为：白细胞总数明显增高，通常大于30×10^9/L。半数患者大于100×10^9/L。中性粒细胞明显增多，可见各阶段粒细胞，以中性中幼粒，晚幼粒细胞居多，原始粒细胞小于等于10％，通常为1％～5％，嗜酸和嗜碱粒细胞亦增多。初诊时约有50％患者血小板增高，少数可大于1000×10^9/L。红细胞和血红蛋白一般在正常范围，若出现血红蛋白减低，血小板计数明显升高或降低，则提示疾病向加速期或急变期转化。7. 历史相关研究: 急性髓系白血病（AML）是造血干细胞恶性克隆性疾病。在AML的诊断、治疗以及判断预后的过程中，基因异常是一项重要指标。随着基因检测技术的不断进步，越来越多与AML发生相关的基因被人们发现，并且这些基因在指导预后方面有重要意义。常见和急性髓系白血病相关的基因突变有: 1）RUNX1-RUNX1T1：8号染色体和21号染色体易位[t(8;21)(q22;q22)]是RUNX1/RUNX1T1融合基因产生的基础。RUNX1/RUNX1T1融合基因与CBFB-MYH11融合基因导致的AML病因、临床特征相似，被合称为CBF-AML。核心结合因子（CBF）在造血干细胞的产生以及造血过程中起到重要作用，RUNX1编码CBF中的ɑ亚基，该亚基负责与DNA直接结合。因此RUNX1/RUNX1T1融合基因的产生会破坏CBF的功能，导致髓系分化阻断并最终导致白血病。2）CBFB-MYH11：CBFB-MYH11融合基因是染色体重排的结果，较常见inv(16)(p13.1q22)，较少见的类型为t(16;16)(p13.1;q22)。该融合基因与M4型AML发生相关，其特征为存在骨髓单核细胞母细胞和非典型嗜酸性粒细胞。小鼠模型表明，CBFB-MYH11融合基因可以破坏核心结合因子（CBF）的功能，导致髓系分化阻断并最终导致白血病。虽然单纯CBFB-MYH11的表达不足以导致白血病的发生，但CBFB-MYH11和其他突变的结合可以特异性地导致髓系白血病的发展。3）NPM1：核磷蛋白1（nucleophosmin 1，NPM1）属于核磷蛋白家族，是一种广泛表达的磷蛋白，能在核仁、核质和胞质之间不断穿梭。该基因位于5q35，包含12个外显子，编码3种核磷蛋白亚型。NPM1主要有4种功能：a 参与核糖体生物合成；b 维持基因的稳定性；c 依赖p53的应激反应；d 通过ARF-p53的相互作用从而调控生长抑制途径。4）CEBPA：CCAAT增强子结合蛋白α基因（CCAAT/en－hancer binding protein α，CEBPA）属亮氨拉链转录因子家族，位于染色体19q13。CEBPA突变会上调造血干细胞归巢和粒细胞分化的基因，下调参与调控造血细胞增殖的信号分子和转录因子的基因，阻碍DNA从G1期向S期演变，且诱导晚期造血细胞成熟，导致白血病发生。CEBPA基因突变发生率在成人AML患者中占5%-14%。突变可分为双突变和单突变，N端移码突变和C端框内突变同时存在即双突变较多见，而单杂合子突变不常见。国内外的多项研究表明双突变的预后良好，无论是CR率及维持化疗后的总CR率CEBPA双突变组均明显高于CEBPA单体突变组及CEBPA阴性组，且中位OS（60个月）和中位EFS（53个月）较其余两组均明显延长。5）MLLT3-KMT2A：MLLT3-KMT2A融合基因是t(9;11)(p21.3;q23.3)染色体易位形成的，其中赖氨酸甲基转移酶2A即KMT2A（旧称MLL）基因突变在AML中较常见，发生率约为10%。其中KMT2A与MLLT3（也被称为AF9或LTG9）基因融合是其中最常见的类型。因相较其他类型KMT2A融合基因的AML，MLLT3-KMT2A融合基因阳性的AML预后差异明显，因此将其单独列出。5）DEK-NUP214：DEK-NUP214融合基因由t(6;9)(p22.3；q34.1)引起，与大概1%的AML发生相关。Carl Sandén等人的研究发现DEK-NUP214基因主要影响细胞增殖过程，通过上调雷帕霉素复合物1(mTORC1)的活性来促进细胞增殖，使用雷帕霉素受体抑制剂治疗对抑制此类白血病细胞增殖有一定效果。DEK-NUP214融合基因阳性的AML通常预后不佳，Slovak所做队列研究显示该类型AML（包括儿童及成人）CR率仅有65%，中位OS仅有13.5个月，中位DFS仅有9.9个月。6）KMT2A：KMT2A基因（也称为MLL基因），位于11q23，编码组蛋白H3赖氨酸4甲基转移酶。KMT2A基因重排发生于大约3%-7%的成人初发AML。KMT2A编码一种组蛋白甲基转移酶，它在胚胎发育和造血过程中对基因表达的维持有重大作用。KMT2A基因易位会产生嵌合的KMT2A融合蛋白，直接与DNA结合并上调基因转录，导致下游KMT2A靶点的异常表达，包括HOX基因等，从而导致AML的发生。问：请基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、相关的基因突变、常用的治疗手段，以及当前已上市和正在临床阶段的药物清单。答：",
    "If I have 100 million dollars, what kinds of projects should I invest to maximize my benefits in background of a growing number of artificial intelligence technologies?",
    "What is OpenVINO?",
};

namespace {

struct Args {
    std::string ov_model_path = "openvino_model.xml";
    std::string token_model_path = "tokenizer.xml";
    std::string detoken_model_path = "detokenizer.xml";
    std::string device = "GPU";
    bool reduce_logits = false;
    bool do_sample = false;
    int top_k = 0;
    float top_p = 0.7;
    float temp = 0.95;
    float repeat_penalty = 1.0;
    int output_fixed_len = 0;
};

static void usage(const std::string& prog) {
    std::cout << "Usage: " << prog << " [options]\n"
        << "\n"
        << "options:\n"
        << "  -h, --help              show this help message and exit\n"
        << "  -m, --model PATH        Chatglm OpenVINO model path (default: openvino_model.xml)\n"
        << "  -token PATH             Tokenizer model path (default: tokenizer.xml)\n"
        << "  -detoken PATH           DeTokenizer model path (default: detokenizer.xml)\n"
        << "  -d, --device            Device (default: GPU)\n"
        << "  --reduce_logits         Reduce_logits (default: False)\n"
        << "  --do_sample             Search (default: False)\n"
        << "  --top_k N               top-k sampling (default: 0)\n"
        << "  --top_p N               top-p sampling (default: 0.7)\n"
        << "  --temp N                temperature (default: 0.95)\n"
        << "  --repeat_penalty N      penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)\n"
        << "  --output_fixed_len N    set output fixed lenth (default: 0, output lenth is determined by the model)\n";
}

static Args parse_args(const std::vector<std::string>& argv) {
    Args args;

    for (size_t i = 1; i < argv.size(); i++) {
        const std::string& arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            exit(EXIT_SUCCESS);
        }
        else if (arg == "-m" || arg == "--model") {
            args.ov_model_path = argv[++i];
        }
        else if (arg == "-token") {
            args.token_model_path = argv[++i];
        }
        else if (arg == "-detoken") {
            args.detoken_model_path = argv[++i];
        }
        else if (arg == "-d" || arg == "--device") {
            args.device = argv[++i];
        }
        else if (arg == "--reduce_logits") {
            args.reduce_logits = true;
        }
        else if (arg == "--do_sample") {
            args.do_sample = true;
        }
        else if (arg == "--top_k") {
            args.top_k = std::stoi(argv[++i]);
        }
        else if (arg == "--top_p") {
            args.top_p = std::stof(argv[++i]);
        }
        else if (arg == "--temp") {
            args.temp = std::stof(argv[++i]);
        }
        else if (arg == "--repeat_penalty") {
            args.repeat_penalty = std::stof(argv[++i]);
        }
        else if (arg == "--output_fixed_len") {
            args.output_fixed_len = std::stoi(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    return args;
}

static Args parse_args(int argc, char** argv) {
    std::vector<std::string> argv_vec;
    argv_vec.reserve(argc);

#ifdef _WIN32
    LPWSTR* wargs = CommandLineToArgvW(GetCommandLineW(), &argc);

    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(converter.to_bytes(wargs[i]));
    }

    LocalFree(wargs);
#else
    for (int i = 0; i < argc; i++) {
        argv_vec.emplace_back(argv[i]);
    }
#endif

    return parse_args(argv_vec);
}

int64_t get_out_token_id(const std::vector<int>& input_ids, float* logits, size_t vocab_size, Args args) {
    int64_t out_token;

    // logits pre-process
    if (args.repeat_penalty != 1.f) {
        sampling_repetition_penalty(logits, logits + vocab_size, input_ids, args.repeat_penalty);
    }

    if (args.do_sample)
    {
        if (args.temp > 0) {
            sampling_temperature(logits, logits + vocab_size, args.temp);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, logits[i]);
        }

        // top_k sampling
        if (0 < args.top_k && args.top_k < (int)token_scores.size()) {
            sampling_top_k(token_scores.data(), token_scores.data() + args.top_k,
                token_scores.data() + token_scores.size());
            token_scores.resize(args.top_k);
        }

        // top_p sampling
        if (0.f < args.top_p && args.top_p < 1.f) {
            auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), args.top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(logits, logits + token_scores.size());
        out_token = token_scores[dist(gen)].id;
    }
    else {
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }

    return out_token;
}

std::pair<ov::Tensor, ov::Tensor> tokenize(ov::InferRequest& tokenizer, std::string && prompt) {
    constexpr size_t BATCH_SIZE = 1;
    tokenizer.set_input_tensor(ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt});
    tokenizer.infer();
    return {tokenizer.get_tensor("input_ids"), tokenizer.get_tensor("attention_mask")};
}

std::string detokenize(ov::InferRequest& detokenizer, std::vector<int64_t>& tokens) {
    constexpr size_t BATCH_SIZE = 1;
    detokenizer.set_input_tensor(ov::Tensor{ov::element::i64, {BATCH_SIZE, tokens.size()}, tokens.data()});
    detokenizer.infer();
    return detokenizer.get_output_tensor().data<std::string>()[0];
}

// The following reasons require TextStreamer to keep a cache of previous tokens:
// detokenizer removes starting ' '. For example detokenize(tokenize(" a")) == "a",
// but detokenize(tokenize("prefix a")) == "prefix a"
// 1 printable token may consist of 2 token ids: detokenize(incomplete_token_idx) == "�"
struct TextStreamer {
    ov::InferRequest detokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;

    void put(int64_t token) {
        token_cache.push_back(token);
        std::string text = detokenize(detokenizer, token_cache);
        if (!text.empty() && '\n' == text.back()) {
            // Flush the cache after the new line symbol
            std::cout << std::string_view{text.data() + print_len, text.size() - print_len};
            token_cache.clear();
            print_len = 0;
            return;
        }
        if (text.size() >= 3 && text.compare(text.size() - 3, 3, "�") == 0) {
            // Don't print incomplete text
            return;
        }
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << std::flush;
        print_len = text.size();
    }

    void end() {
        std::string text = detokenize(detokenizer, token_cache);
        std::cout << std::string_view{text.data() + print_len, text.size() - print_len} << '\n';
        token_cache.clear();
        print_len = 0;
    }
};

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

/*@brief Insert slice transformation matches following graph, start from logits (Results) to search along root->parent-> grandparent node,
 * then insert slice between Reshape (grandparent node) and Matmul to keep only last dim of matmul first input, first input shape reduced
 * from [1, seq_len, 4096] to [1, 1,4096]. Therefore, after graph transformation, we can reduce matmul computation
 * from [1, seq_len, 4096] * [1, 4096, 151936] = [1, seq_len, 151936] to [1,1,4096]*[4096,151936] = [1,1,151936]
 *
 * Original graph
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
 *
 * Modified graph after insert slice:
 *
 *         +----------+            +----------+
 *         |  Reshape |            | Constant |
 *         +----------+            +----------+
 *              |                       |
 *         +----------+                 |
 *         |  Slice   |                 |
 *         +----------+                 |
 *              |                       |
 *              -----------    ----------
 *                        |    |
 *                        v    v
 *                      +--------+
 *                      | MatMul |
 *                      +--------+
 *                          |
 *                          v
 *                     +----------+
 *                     |  logits  |
 *                     +----------+
*/

class InsertSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertSlice", "0");
    explicit InsertSlice() {
        auto label = ov::pass::pattern::wrap_type<ov::op::v0::Result>();
        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
            auto root = std::dynamic_pointer_cast<ov::op::v0::Result>(m.get_match_root());
            if (!root) {
                return false;
            }
            std::string root_name = root->get_friendly_name();
            if (root->get_output_partial_shape(0).size() == 3) {
                std::cout << "Find target root node name: " << root_name << "\n";
                auto parent = root->input_value(0).get_node_shared_ptr();
                std::cout << "Find parent node name: " << parent->get_friendly_name() << "\n";

                //llama2
                auto grand_parent = parent->input_value(0).get_node_shared_ptr();
                std::cout << "Find grandparent node name: " << grand_parent->get_friendly_name() << "\n";

                ov::Output<ov::Node> grand_parent_output = parent->get_input_source_output(0); // parent->get_input_source_output(0);
                std::set<ov::Input<ov::Node>> consumers = grand_parent_output.get_target_inputs();
                auto partial_shape = grand_parent_output.get_partial_shape().get_min_shape();
                int32_t dims = static_cast<int32_t>(partial_shape[2]);
		    
                std::vector<int32_t> start_v = { 0, -1, 0 };
                std::vector<int32_t> stop_v = { 1, -2, dims };
                std::vector<int32_t> step_v = { 1, -1, 1 };

                std::cout << "Original reshape node output shape:" << grand_parent_output.get_partial_shape() << std::endl;
                auto starts = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    start_v);
                auto stop = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    stop_v);
                auto step = ov::op::v0::Constant::create(ov::element::i32,
                    ov::Shape{ 3 },
                    step_v);
                auto slice = std::make_shared<ov::opset13::Slice>(grand_parent, starts, stop, step); //data, starts, ends, steps
                std::cout << "After insert slice node, output shape" << slice->output(0).get_partial_shape() << std::endl;
                for (auto consumer : consumers) {
                    consumer.replace_source_output(slice->output(0));
                }
                register_new_node(slice);
            }

            return true;
            };
        // Register pattern with Parameter operation as a pattern root node
        auto m = std::make_shared<ov::pass::pattern::Matcher>(label, "InsertSlice");
        // Register Matcher
        register_matcher(m, callback);
    }
};

}

int main(int argc, char* argv[]) try {

    Args args = parse_args(argc, argv);

    std::cout << ov::get_openvino_version() << std::endl;
	
    ov::Core core;
    core.add_extension(USER_OV_EXTENSIONS_PATH);  // USER_OV_EXTENSIONS_PATH is defined in root CMakeLists.txt
    auto startTime = Time::now();
    ov::InferRequest tokenizer = core.compile_model(args.token_model_path, "CPU").create_infer_request();
    auto input_ids = tokenizer.get_tensor("input_ids");
    auto attention_mask = tokenizer.get_tensor("attention_mask");
    ov::InferRequest detokenizer = core.compile_model(args.detoken_model_path, "CPU").create_infer_request();
    auto duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Load chatglm tokenizer took " << duration_ms << " ms" << std::endl;
    std::string device = args.device;
    constexpr size_t BATCH_SIZE = 1;
    size_t convert_model;

    if (args.reduce_logits){
        convert_model = 1;
    }
    else {
        convert_model = 0;
    }
	
    ov::AnyMap device_config = {};
    if (device.find("CPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::hint::scheduling_core_type.name()] = ov::hint::SchedulingCoreType::PCORE_ONLY;
        device_config[ov::hint::enable_hyper_threading.name()] = false;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }

    if (device.find("GPU") != std::string::npos) {
        device_config[ov::cache_dir.name()] = "llm-cache";
        device_config[ov::intel_gpu::hint::queue_throttle.name()] = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
        device_config[ov::intel_gpu::hint::queue_priority.name()] = ov::hint::Priority::MEDIUM;
        device_config[ov::intel_gpu::hint::host_task_priority.name()] = ov::hint::Priority::HIGH;
        device_config[ov::hint::enable_cpu_pinning.name()] = true;
        device_config[ov::enable_profiling.name()] = false;
    }

    double total_time = 0;
    int count = 0;
    double first_time;
    
    // Read OpenVINO Model
    if (1 == convert_model) {
        startTime = Time::now();
        std::shared_ptr<ov::Model> model = core.read_model(args.ov_model_path);
        duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "Read chatglm Model took " << duration_ms << " ms" << std::endl;

        std::cout << "######## [Model Graph Optimization] Step 2: Insert slice node after reshape to reduce logits operation ########\n";
        ov::pass::Manager manager;
        manager.register_pass<InsertSlice>();
        manager.run_passes(model);

        std::string modifiled_file = std::regex_replace(args.ov_model_path, std::regex("openvino_model"), "modified_openvino_model");
        std::cout << "Save modified model in " << modifiled_file << "\n";
        ov::serialize(model, modifiled_file);

        ov::CompiledModel compilemodel = core.compile_model(modifiled_file, device, device_config);

        return 0;
    }

    //Compile model
    startTime = Time::now();
    ov::CompiledModel compilemodel = core.compile_model(args.ov_model_path, device, device_config);
    ov::InferRequest ireq = compilemodel.create_infer_request();
    duration_ms = get_duration_ms_until_now(startTime);
    std::cout << "Compile LLM model took " << duration_ms << " ms" << std::endl;
    /*
    auto supported_properties = compilemodel.get_property(ov::supported_properties);
    std::cout << "Model:" << std::endl;
    for (const auto& cfg : supported_properties) {
        if (cfg == ov::supported_properties)
            continue;
        auto prop = compilemodel.get_property(cfg);
        if (cfg == ov::device::properties) {
            auto devices_properties = prop.as<ov::AnyMap>();
            for (auto& item : devices_properties) {
                std::cout << "  " << item.first << ": " << std::endl;
                for (auto& item2 : item.second.as<ov::AnyMap>()) {
                    OPENVINO_SUPPRESS_DEPRECATED_START
                        if (item2.first == ov::supported_properties ) 
                            continue;
                    OPENVINO_SUPPRESS_DEPRECATED_END
                        std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
                }
            }
        }
        else {
            std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
        }
    } */
 
    auto model_inputs = compilemodel.inputs();
    auto inputs = compilemodel.inputs();
    TextStreamer text_streamer{ std::move(detokenizer) };

    for (std::string input_text : sentences) {
        total_time = 0;
        count = 0;
        auto prompt_text = "[INST] " + input_text + " [/INST]"; //llama2-chat
        std::cout << " #### sentence: index " << prompt_text << std::endl;
        tokenize(tokenizer, prompt_text.c_str());
        input_ids = tokenizer.get_tensor("input_ids");
        attention_mask = tokenizer.get_tensor("attention_mask");
        std::cout << "input lenghth " << input_ids.get_size() << std::endl;
	    
        std::vector<int> output_ids;
        output_ids.reserve(input_ids.get_size());
        for (size_t idx = 0; idx < input_ids.get_size(); ++idx) {
            output_ids.emplace_back(((int)input_ids.data<const int64_t>()[idx]));
        }
        
        ireq.set_tensor("input_ids", input_ids);
        ireq.set_tensor("attention_mask", attention_mask);
        ireq.get_tensor("position_ids").set_shape(input_ids.get_shape());
        std::iota(ireq.get_tensor("position_ids").data<int64_t>(), ireq.get_tensor("position_ids").data<int64_t>() + ireq.get_tensor("position_ids").get_size(), 0);
        ireq.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
        ireq.get_tensor("beam_idx").data<int32_t>()[0] = 0;

	for (auto &&state : ireq.query_state()){
            state.reset();
        }

        startTime = Time::now();
        ireq.infer();
        duration_ms = get_duration_ms_until_now(startTime);
        std::cout << "First token took " << duration_ms << " ms" << std::endl;
        first_time = duration_ms;

        size_t vocab_size = ireq.get_tensor("logits").get_shape().back();

        //float* logits = ireq.get_tensor("logits").data<float>();
        float* logits = ireq.get_tensor("logits").data<float>() + (input_ids.get_size() - 1) * vocab_size;

        int64_t out_token = get_out_token_id(output_ids, logits, vocab_size, args);
        output_ids.emplace_back(((int)out_token));

        ireq.get_tensor("input_ids").set_shape({ BATCH_SIZE, 1 });
        ireq.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

        constexpr int64_t SPECIAL_EOS_TOKEN = 2;  // There's no way to extract the value from the detokenizer for now
        while (true) {  //(out_token != SPECIAL_EOS_TOKEN)
            startTime = Time::now();
            ireq.get_tensor("input_ids").data<int64_t>()[0] = out_token;
            ireq.get_tensor("attention_mask").set_shape({ BATCH_SIZE, ireq.get_tensor("attention_mask").get_shape()[1] + 1 });
            std::fill_n(ireq.get_tensor("attention_mask").data<int64_t>(), ireq.get_tensor("attention_mask").get_size(), 1);
            ireq.get_tensor("position_ids").data<int64_t>()[0] = ireq.get_tensor("attention_mask").get_size() - 2;
            
            ireq.start_async();
            ireq.wait();
            duration_ms = get_duration_ms_until_now(startTime);
            count += 1;
            total_time += duration_ms;

            text_streamer.put(out_token);
            logits = ireq.get_tensor("logits").data<float>();

            out_token = get_out_token_id(output_ids, logits, vocab_size, args);
            output_ids.emplace_back(((int)out_token));

            if (args.output_fixed_len > 0) {
                if(count >= (args.output_fixed_len - 1))
                    break;
            } 
            else {
                if (out_token == SPECIAL_EOS_TOKEN) {
                    break;
                }
            }
        }
        
        text_streamer.end();

        if (count > 0) {
            std::cout << "Other Avg inference took total " << total_time << " ms token num " << count << " first " << first_time << " ms " << " avg " << total_time / (count) << " ms" << std::endl;
        }
    }
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return 1;
}
