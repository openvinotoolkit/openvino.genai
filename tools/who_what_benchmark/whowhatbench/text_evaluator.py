from typing import Any, Union

import os
import pandas as pd
from tqdm import tqdm

from .registry import register_evaluator, BaseEvaluator
from .whowhat_metrics import TextDivergency, TextSimilarity

default_data = {
    "en": {
        "prompts": [
            "As enticing as quantum entanglement is, it also raises philosophical questions regarding the nature of reality. The phenomenon challenges our conventional understanding of causality and the separability of objects. If entangled particles can instantaneously affect each other's states over vast distances, what does this imply about the structure of spacetime? Some interpretations of quantum mechanics, such as the many-worlds interpretation, suggest that every possible outcome occurs in a separate, branching universe, thereby preserving locality while explaining entanglement.",
            "At its core, quantum entanglement occurs when two or more particles become interconnected in such a way that the state of one particle instantaneously influences the state of another, regardless of the distance separating them. This interdependence persists even if the particles are light-years apart, defying classical intuitions about the separability of objects. To understand how this phenomenon arises, we must explore the principles of superposition and measurement in quantum mechanics.",
            "Despite significant advances, there remains much to explore regarding the nature and applications of quantum entanglement. Scientists are actively researching how to harness and manipulate entangled states more effectively. As technology improves, we may unlock new ways to utilize this phenomenon for practical applications, such as improved GPS systems, advanced sensors, and even insights into the fabric of spacetime itself.",
            "Entanglement has implications far beyond theoretical physics; it plays a crucial role in the burgeoning field of quantum computing. Classical computers process information in bits that represent either a 0 or a 1. Quantum computers, however, leverage the principles of quantum mechanics to manipulate quantum bits, or qubits, which can represent both 0 and 1 simultaneously due to superposition. Entangled qubits can work together to perform complex calculations at speeds unachievable by classical computers, potentially revolutionizing fields like cryptography, drug discovery, and complex system simulations.",
            "In a small town nestled between rolling hills, there existed a peculiar library known as The Whispering Pages . Its walls were filled with ancient tomes and modern texts alike, each with a story eager to be told . The townsfolk often spoke of the library's magic, claiming that the books would whisper secrets to anyone willing to listen . Many dismissed it as mere folklore, but for twelve-year-old Mira, it was a place of wonder and discovery . Every Saturday, Mira would visit The Whispering Pages after finishing her chores . She had a routine: first, she would greet Mr . Tobias, the elderly librarian with twinkling eyes and a white mustache that danced when he smiled . “Welcome back, Mira,” he would say, as if he had been waiting all week for her return .",
            "Mira would then make her way to her favorite corner, a snug little nook bathed in sunlight, where a massive oak tree outside provided shade and created a cozy atmosphere . On one such Saturday, as Mira settled in with a book about mythical creatures, she began to hear a faint voice . She looked around, but the library was empty except for Mr . Tobias, who was busy sorting through a stack of new arrivals . The voice became clearer, and to her surprise, it was coming from the book itself . \"Find the heart that beats beneath the stones,\" it murmured . Intrigued, Mira leaned closer . She flipped through the pages, and as she did, she noticed something unusual—a map drawn in the margins . It appeared to lead beyond the library, out into the hills that surrounded the town . Her curiosity piqued, she carefully removed the book from the shelf and tucked it under her arm, deciding that after her visit, she would investigate further .",
            "In conclusion, quantum entanglement is a captivating area of study that has reshaped our understanding of the physical universe. It is a testament to the oddities of quantum mechanics, where particles can be deeply connected regardless of the distances that separate them. With ongoing research and technological advancements, the concept of entanglement continues to inspire new theories and applications, offering a glimpse into a future where quantum systems may revolutionize how we process information and interact with the world around us. As we delve deeper into the quantum realm, we uncover not just the intricacies of particles and forces but also fundamental truths about the nature of reality itself.",
            "In quantum mechanics, particles such as electrons or photons exist in a state of superposition. This means they do not have definite properties until measured. For example, an electron can simultaneously have a spin of \"up\" and \"down\" until an observation is made. When two particles are entangled, their superposed states are linked. If one particle is measured and found to have a specific property, the other particle’s state is determined instantaneously—the spin of the second particle will be opposite that of the first, regardless of the distance between them.",
            "Moreover, quantum entanglement is a critical component of quantum communication. It enables secure transmission of information through techniques like quantum key distribution (QKD). In QKD, two parties can share a secret key using entangled particles. Any attempt by an eavesdropper to intercept or measure the particles will disturb their states, revealing the presence of an unauthorized observer. This technology promises a significant advancement in data security, offering virtually unbreakable encryption.",
            "Once home, Mira spread out her findings across her bedroom floor . The map was rudimentary, marked with simple symbols: a sun, a tree, and an ominous 'X' at the end . It felt like a treasure map, and Mira's imagination began to race . After her parents went to bed, she gathered supplies: a flashlight, a notebook, and a snack for the journey . With her heart racing at the thought of adventure, she headed out into the cool night . The moon illuminated her path as Mira made her way up the hillside, following the map's directions . The night was quiet, with only the sound of rustling leaves and the distant hoot of an owl . As she climbed higher, she felt a growing sense of purpose . “The heart that beats beneath the stones,” she muttered, trying to decipher what the words could mean . After some time, she arrived at a clearing where the ground was carpeted with moss and dotted with smooth stones . The map indicated that she needed to look closely . Mira knelt down to inspect the area and, just as she was about to give up, she heard a soft thump, like the beat of a drum . Surprised, she looked around and found a particularly large stone slightly displaced from the others . The crystal became her talisman, reminding her of her promise and the magic of storytelling—a bridge between the ordinary and the extraordinary, where dreams take flight and every book waited to be opened . ",
            "Quantum entanglement is one of the most intriguing phenomena in the realm of quantum mechanics, a branch of physics that describes the behavior of matter and energy at the smallest scales. Developed in the early 20th century, quantum mechanics fundamentally altered our perception of the universe. Unlike classical physics, which dictates that particles have defined positions and velocities, quantum mechanics introduces a level of uncertainty and non-locality. One of the cornerstones of this theory is the concept of entanglement, which Albert Einstein famously referred to as \"spooky action at a distance.\"",
            "### The Fascinating World of Bioluminescence  #### Introduction Bioluminescence is a natural phenomenon that occurs in various organisms, characterized by the ability to emit light . This incredible adaptation can be found in a range of living beings, including certain species of fungi, bacteria, and marine animals . The light produced can serve various purposes such as predation, communication, and camouflage . This article explores the mechanisms, examples, and ecological significance of bioluminescence, shedding light on its role in the natural world .",
            "The process of bioluminescence involves a biochemical reaction between a light-emitting molecule known as luciferin and an enzyme called luciferase . This reaction occurs within specialized cells or organelles and typically requires oxygen . The specific structure of luciferin varies among different organisms, leading to a wide range of colors emitted, from blue and green to red and yellow . The basic biochemical reaction can be summarized as follows:  1 . **Formation of Luciferin-Oxygen Complex**: When luciferin reacts with oxygen in the presence of luciferase, it forms an unstable complex . 2 .",
            "The implications of quantum entanglement extend beyond fundamental physics. They intersect with various fields, including thermodynamics, information theory, and even biology. Researchers are exploring the possibility that quantum entanglement plays a role in biological processes, such as photosynthesis and avian navigation. For example, certain birds are thought to navigate using quantum coherence in their eyes. This intriguing intersection of quantum phenomena and biological systems suggests that entanglement may be a universal principle, manifesting in diverse contexts across nature.",
            "The study of entanglement has also led to the exploration of quantum teleportation—the process of transferring quantum states from one location to another without physically moving the particle itself. By creating a pair of entangled particles, where one remains at point A and the other is sent to point B, the state of the particle at point A can be \"teleported\" to point B through a classical communication channel. This concept is not merely science fiction; researchers have successfully demonstrated teleportation of quantum states in laboratory settings, paving the way for potential advancements in quantum networks.",
        ],
    },
    "cn": {
        "prompts": [
            "马克吐温是谁?",
            "谁是威廉-莎士比亚?",
            "阿加莎-克里斯蒂是谁?",
            "芭芭拉-卡特兰是谁?",
            "丹妮尔-斯蒂尔是谁?",
            "谁是哈罗德-罗宾斯?",
            "乔治-西默农是谁?",
            "伊妮德-布莱顿是谁?",
            "西德尼-谢尔顿是谁?",
            "鸟山明是谁?",
            "谁是列夫-托尔斯泰?",
            "亚历山大-普希金是谁?",
            "斯蒂芬-金是谁?",
            "C++是什么?",
            "Python是什么?",
            "什么是 Java?",
            "JavaScript是什么?",
            "什么是 Perl?",
            "什么是 OpenCV?",
            "谁是最著名的作家?",
            "谁是最有名的发明家?",
            "谁是最著名的数学家?",
            "最著名的作曲家是谁?",
            "谁是最有名的程序员?",
            "谁是最著名的运动员?",
            "谁是最著名的古希腊科学家?",
            "蓝色和黄色混合会得到什么颜色?",
        ],
    },
}


@register_evaluator(
    "text"
)
class TextEvaluator(BaseEvaluator):
    def __init__(
        self,
        base_model: Any = None,
        tokenizer: Any = None,
        gt_data: str = None,
        test_data: Union[str, list] = None,
        metrics="similarity",
        similarity_model_id: str = "sentence-transformers/all-mpnet-base-v2",
        max_new_tokens=128,
        crop_question=True,
        num_samples=None,
        language="en",
        gen_answer_fn=None,
        generation_config=None,
        generation_config_base=None,
        seqs_per_request=None,
        use_chat_template=None,
    ) -> None:
        assert (
            base_model is not None or gt_data is not None
        ), "Text generation pipeline for evaluation or ground trush data must be defined"

        self.test_data = test_data
        self.metrics = metrics
        self.max_new_tokens = max_new_tokens
        self.tokenizer = tokenizer
        self._crop_question = crop_question
        self.num_samples = num_samples
        self.generation_config = generation_config
        self.generation_config_base = generation_config
        self.seqs_per_request = seqs_per_request
        self.generation_fn = gen_answer_fn
        self.use_chat_template = use_chat_template
        if self.generation_config is not None:
            assert self.seqs_per_request is not None

        # Take language from the base model if provided
        self.language = language

        if base_model:
            self.gt_data = self._generate_data(
                base_model, gen_answer_fn, generation_config=generation_config
            )
        else:
            self.gt_data = pd.read_csv(gt_data, keep_default_na=False)

        # Take language ground truth if no base model provided
        if self.language is None and "language" in self.gt_data.columns:
            self.language = self.gt_data["language"].values[0]

        self.similarity = None
        self.divergency = None
        if "similarity" in self.metrics:
            self.similarity = TextSimilarity(similarity_model_id)
        if "divergency" in self.metrics:
            assert tokenizer is not None
            self.divergency = TextDivergency(tokenizer)

        self.last_cmp = None

    def get_generation_fn(self):
        return self.generation_fn

    def score(self, model_or_data, gen_answer_fn=None, **kwargs):
        if isinstance(model_or_data, str) and os.path.exists(model_or_data):
            predictions = pd.read_csv(model_or_data, keep_default_na=False)
        else:
            predictions = self._generate_data(model_or_data, gen_answer_fn, self.generation_config)
        self.predictions = predictions

        all_metrics_per_prompt = {}
        all_metrics = {}

        if self.similarity:
            metric_dict, metric_per_question = self.similarity.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        if self.divergency:
            metric_dict, metric_per_question = self.divergency.evaluate(
                self.gt_data, predictions
            )
            all_metrics.update(metric_dict)
            all_metrics_per_prompt.update(metric_per_question)

        self.last_cmp = all_metrics_per_prompt
        self.last_cmp["prompts"] = predictions["prompts"].values
        self.last_cmp["source_model"] = self.gt_data["answers"].values
        self.last_cmp["optimized_model"] = predictions["answers"].values
        self.last_cmp = pd.DataFrame(self.last_cmp)
        self.last_cmp.rename(columns={"prompts": "prompt"}, inplace=True)

        return pd.DataFrame(all_metrics_per_prompt), pd.DataFrame([all_metrics])

    def worst_examples(self, top_k: int = 5, metric="similarity"):
        assert self.last_cmp is not None

        if metric in ["SDT", "SDT norm"]:
            res = self.last_cmp.nlargest(top_k, metric)
        else:
            res = self.last_cmp.nsmallest(top_k, metric)

        res = list(row for idx, row in res.iterrows())

        return res

    def _generate_data(self, model, gen_answer_fn=None, generation_config=None):
        def default_gen_answer(model, tokenizer, prompt, max_new_tokens, crop_question, use_chat_template=False):
            if use_chat_template:
                message = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
                tokens = model.generate(inputs, do_sample=False, max_new_tokens=max_new_tokens)
                if crop_question:
                    tokens = tokens[:, inputs.shape[-1]:]
                res = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
                return res
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                if crop_question:
                    tokens = tokens[:, inputs["input_ids"].shape[-1] :]
                return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

        gen_answer_fn = gen_answer_fn or default_gen_answer

        if self.test_data:
            if isinstance(self.test_data, str):
                data = pd.read_csv(self.test_data)
            else:
                if isinstance(self.test_data, dict):
                    assert "prompts" in self.test_data
                    data = dict(self.test_data)
                else:
                    data = {"prompts": list(self.test_data)}
                data = pd.DataFrame.from_dict(data)
        else:
            data = pd.DataFrame.from_dict(default_data[self.language])

        prompt_data = data["prompts"]

        answers = []
        prompts = (
            prompt_data.values
            if self.num_samples is None
            else prompt_data.values[: self.num_samples]
        )

        if generation_config is None:
            for p in tqdm(prompts, desc="Evaluate pipeline"):
                answers.append(
                    gen_answer_fn(
                        model,
                        self.tokenizer,
                        p,
                        self.max_new_tokens,
                        self._crop_question,
                        self.use_chat_template
                    )
                )
        else:
            with tqdm(total=len(prompt_data.values)) as progress_bar:
                batch = []
                for p_idx, p in enumerate(prompt_data.values):
                    progress_bar.update(1)
                    batch.append(p)
                    if (
                        len(batch) == self.seqs_per_request
                        or p_idx == len(prompt_data.values) - 1
                    ):
                        ans_batch = model.generate(
                            batch, [generation_config] * len(batch)
                        )
                        for ans in ans_batch:
                            answers.append(ans.m_generation_ids[0])

                        batch.clear()

        res_data = {"prompts": list(prompts), "answers": answers}
        df = pd.DataFrame(res_data)
        df["language"] = self.language

        return df
