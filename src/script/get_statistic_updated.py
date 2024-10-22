import csv
import os
from subprocess import Popen, TimeoutExpired, run, call
from statistics import median

model_path = "/home/efode/repo/openvino.genai/temp/speculative_decoding/customer_test"

models = {
    # model_name: main_model, assistant_model
    "llama" : [
        ("llama-2-7b-chat", "AMD-Llama-135m"),
        ("llama-2-7b-chat", "llama2-68m-chat"),
        ("llama-2-7b-chat", "TinyLlama-1.1B-Chat-v1.0"),
        ("CodeLlama-7b-hf", "AMD-Llama-135M-code"),
    ],
    "opt": [("opt-6.7b", "opt-125m")],
    "phi": [("Phi-3-medium-4k-instruct", "Phi-3-mini-4k-instruct")],
    "qwen": [("qwen1.5-7b", "qwen1.5-0.5b")],
    "starcoder": [("starcoder2-7b", "starcoder2-164m")],
}

element_types = [""]

# num_run = range(0, 3)
num_run = range(0, 3)

# prompt = "Article: While Google and Apple are developing self driving vehicles, Nasa has revealed its vision - and its a lot more fun. Called the Modular Robotic Vehicle, or MRV, it was developed at NASA's Johnson Space Center to show off the technologies that could let man move across other planets. On Earth, however, it also has some neat tricks - from the ability to park sideways to being able to spin on the spot, as well as being driven remotely at up to 70mph. Watch the buggy in action below . Unlike a normal car, the buggy has no mechanical linkages to the propulsion, steering, or brake actuators, the driver of an MRV relies completely on control inputs being converted to electrical signals and then transmitted by wires to the vehicle's motors. A turn of the steering wheel, for instance, is recorded by sensors and sent to computers at the rear of the vehicle. These computers interpret that signal and instruct motors at one or all four of the wheels to move at the appropriate rate, causing the vehicle to turn as commanded. Nasa says the vehicles was buit 'in order to advance technologies that have applications for future vehicles both in space and on Earth.' 'With seating for two people, MRV is a fully electric vehicle well-suited for busy urban environments, it says. MRV is driven by four independent wheel modules called e-corners. Each e-corner consists of a redundant steering actuator, a passive trailing arm suspension, an in-wheel propulsion motor, and a motor-driven friction braking system. Each e-corner can be controlled independently and rotated ±180 degrees about its axis. This allows for a suite of driving modes allowing MRV to maneuver unlike any traditional vehicle on the road. In addition to conventional front two wheel steering, the back wheels can also articulate allowing for turning radiuses as tight as zero. The driving mode can be switched so that all four wheels point and move in the same direction achieving an omni-directional, crab-like motion. This makes a maneuver such as parallel parking as easy as driving next to an available spot, stopping, and then operating sideways to slip directly in between two cars. 'This two-seater vehicle was designed to meet the growing challenges and demands of urban transportation,' said Mason Markee, also with Johnson. 'The MRV would be ideal for daily transportation in an urban environment with a designed top speed of 70 km/hr and range of 100 km of city driving on a single charge of the battery. 'The size and maneuverability of MRV gives it an advantage in navigating and parking in tight quarters.' With a designed top speed of around 70 km/hr, test driving proved to be a bit of fun. Justin Ridley told an Nasa magazine: 'It's like driving on ice but having complete control. 'It's a blast to ride in and even more fun to drive. We've talked about it being like an amusement park ride. 'The 'fun' of driving was not something we tried to design for, just something that came out of the design. 'Once we got it running many of us commented that we had no idea it was going to be able to do the things it does.' 'This work also allowed us to develop some technologies we felt were needed for our future rovers,' said Ridley. The driver controls MRV with a conventional looking steering wheel and accelerator/brake pedal assembly. The driving mode can be switched so that all four wheels point and move in the same direction - perfect for parallel parking. 'These include redundant by-wire systems, liquid cooling, motor technology, advanced vehicle control algorithms. 'We were able to learn a lot about these and other technologies by building this vehicle.' The buggy can also be controlled remotely, and in the future Nasa says this system can be expanded to allow for autonomous driving. The driver controls MRV with a conventional looking steering wheel and accelerator/brake pedal assembly. Both of these interfaces were specially designed to mimic the feel of the mechanical/hydraulic systems that people are used to feeling when driving their own cars. The buggy can also be controlled remotely, and in the future Nasa says this system can be expanded to allow for autonomous driving. 'While the vehicle as a whole is designed around operating in an urban environment, the core technologies are advancements used in many of our robotic systems and rovers,' said Mason. 'Actuators, motor controllers, sensors, batteries, BMS, component cooling, sealing, and software are all examples of technologies that are being devel oped and tested in MRV that will be used in next generation rover systems. 'The technologies developed in MRV have direct application in future manned vehicles undertaking missions on the surface of Earth's moon, on Mars, or even an asteroid. 'Additionally, MRV provides a platform to learn lessons that could drive the next generation of automobiles."\
# "\n\nSummarize the above article in 5 sentence."

prompt = "Tell me something about Russia's history"

sampling_types = [ "greedy" ]
# sampling_types = [ "multinomial", "greedy" ]

speculative_decoding_type = [ "static", "dynamic" ]

gen_len = [ 10 ]
# gen_len = [ 100, 500 ]

devices = [ "CPU" ]

row_desc = [
    "main_model",
    "main_model_type",
    "draft_model",
    "draft_model_type",

    "speculative_decoding_type",
    "sampling_type",
    "generation_len",
    "main_device",
    "draft_device",

    "llm_token_per_second",
    "cb_token_per_second",
    "speculative_decoding_token_per_second",

    "speculative_decoding_iterations",
    "speculative_decoding_acceptanced_tokens",
    "speculative_decoding_acceptanced_tokens, %",
    "speculative_decoding_avg_acceptance_rate, %",

    "draft_model_duration, ms",
    "draft_model_duration, %",
    "main_model_duration, ms",
    "main_model_duration, %",

    "llm_duration, ms",
    "cb_duration, ms",
    "speculative_decoding_duration, ms",

    "llm_vs_speculative_decoding_boost",
    "llm_vs_speculative_decoding_boost, %",
    "cb_vs_speculative_decoding_boost",
    "cb_vs_speculative_decoding_boost, %",
]


codenames = {
    "total_duration": "Total duration, ms: ",
    "draft_duration": "Draft model duration, ms: ",
    "main_duration": "Main model duration, ms: ",
    "draft_duration_percentage": "Draft model duration, %: ",
    "main_duration_percentage": "Main model duration, %: ",
    "token_per_second": "Token per sec:",
    "iterations": "Main model iterations: ",
    "avg_acceptance_rate": "AVG acceptance rate, %: ",
    "accepted_token_cnt": "Accepted tokens by draft model: ",
    "accepted_token_rate": "Accepted token rate, %: "
}

binary_path = "/home/efode/repo/openvino/bin/intel64/Release"

replace = "REPLACE_"
cb_binary_name = "speculative_decoding_lm"
llm_binary_name = f"{replace}_causal_lm"

log_path = "/home/efode/repo/openvino.genai/temp/speculative_decoding/log.log"
csv_path = "/home/efode/repo/openvino.genai/temp/speculative_decoding/results_avg.csv"
csv_path_all = "/home/efode/repo/openvino.genai/temp/speculative_decoding/results_all.csv"

def parse_log():
    duration = 0
    with open(log_path, "r", encoding='UTF-8') as log_file:
        try:
            lines = log_file.readlines()
        except:
            from pathlib import Path
            log_path_ = Path(log_path)
            lines = log_path_.read_text(encoding="ascii", errors="ignore").split("\n")
        for line in lines:
            if codenames['total_duration'] in line:
                duration = float(line.replace(codenames['total_duration'], "").replace("\n", ""))
                break
    os.remove(log_path)
    return duration

def sd_parse_log():
    with open(log_path, "r", encoding='UTF-8') as log_file:
        try:
            lines = log_file.readlines()
        except:
            from pathlib import Path
            log_path_ = Path(log_path)
            lines = log_path_.read_text(encoding="ascii", errors="ignore").split("\n")

    for line in lines:
        if codenames['total_duration'] in line:
            duration = float(line.replace(codenames['total_duration'], "").replace("\n", ""))
        elif codenames['draft_duration'] in line:
            draft_duration = float(line.replace(codenames['draft_duration'], "").replace("\n", ""))
        elif codenames['main_duration'] in line:
            main_duration = float(line.replace(codenames['main_duration'], "").replace("\n", ""))
        elif codenames['draft_duration_percentage'] in line:
            draft_duration_percentage = float(line.replace(codenames['draft_duration_percentage'], "").replace("\n", ""))
        elif codenames['main_duration_percentage'] in line:
            main_duration_percentage = float(line.replace(codenames['main_duration_percentage'], "").replace("\n", ""))
        elif codenames['token_per_second'] in line:
            token_per_second = float(line.replace(codenames['token_per_second'], "").replace("\n", ""))
        elif codenames['iterations'] in line:
            iterations = int(line.replace(codenames['iterations'], "").replace("\n", ""))
        elif codenames['avg_acceptance_rate'] in line:
            avg_acceptance_rate = float(line.replace(codenames['avg_acceptance_rate'], "").replace("\n", ""))
        elif codenames['accepted_token_cnt'] in line:
            accepted_token_cnt = float(line.replace(codenames['accepted_token_cnt'], "").replace("\n", ""))
        elif codenames['accepted_token_rate'] in line:
            accepted_token_rate = float(line.replace(codenames['accepted_token_rate'], "").replace("\n", ""))
        else:
            continue
    os.remove(log_path)
    return (duration, draft_duration, main_duration, draft_duration_percentage, main_duration_percentage, token_per_second, iterations, avg_acceptance_rate, accepted_token_cnt, accepted_token_rate)

with open(csv_path_all, "w", encoding='UTF-8') as csv_all_file:
    csv_writer_all = csv.writer(csv_all_file, dialect="excel")
    csv_writer_all.writerow(row_desc)

    with open(csv_path, "w", encoding='UTF-8') as csv_file:
        csv_writer = csv.writer(csv_file, dialect="excel")
        csv_writer.writerow(row_desc)

        for model_name in models:
            for model_pair in models[model_name]:
                for main_element_type in element_types:
                    for draft_element_type in element_types:
                        for generation_len in gen_len:
                            for sampling in sampling_types:
                                for sd_type in speculative_decoding_type:
                                    avg_llm_duration = []
                                    avg_llm_token_per_sec = 0
                                    avg_cb_duration = []
                                    avg_cb_token_per_sec = 0
                                    avg_sd_duration = []
                                    avg_sd_draft_duration = []
                                    avg_sd_main_duration = []
                                    avg_sd_draft_duration_percentage = []
                                    avg_sd_main_duration_percentage = []
                                    avg_sd_token_per_second = 0
                                    avg_sd_iterations = 0
                                    avg_sd_avg_acceptance_rate = 0
                                    avg_sd_accepted_token_cnt = 0
                                    avg_sd_accepted_token_rat = 0
                                    avg_cd_token_per_sec = 0
                                    avg_sd_accepted_token_rate = 0
                                    avg_llm_boost = 0
                                    avg_llm_boost_ = 0
                                    avg_cb_boost = 0
                                    avg_cb_boost_ = 0

                                    for run_id in num_run:
                                        llm_binary_name = llm_binary_name.replace(replace, sampling)
                                        llm_exec_path = os.path.join(binary_path, llm_binary_name)
                                        cb_exec_path = os.path.join(binary_path, cb_binary_name)

                                        main_model_path = os.path.join(model_path, model_name, model_pair[0], main_element_type)
                                        draft_model_path = os.path.join(model_path, model_name, model_pair[1], draft_element_type)

                                        if not os.path.exists(main_model_path) or not os.path.exists(draft_model_path):
                                            continue

                                        command_line_llm = f"{llm_exec_path} {main_model_path} \"{prompt}\" {generation_len} > {log_path}"
                                        command_line_cb = f"{cb_exec_path} {main_model_path} none \"{prompt}\" {generation_len} {sampling} {sd_type} > {log_path}"
                                        command_line_sd = f"{cb_exec_path} {main_model_path} {draft_model_path} \"{prompt}\" {generation_len} {sampling} {sd_type} > {log_path}"

                                        llm_duration = 0
                                        llm_token_per_sec = 0
                                        cb_duration = 0
                                        cd_token_per_sec = 0
                                        sd_duration = 0
                                        sd_draft_duration = 0
                                        sd_main_duration = 0
                                        sd_draft_duration_percentage = 0
                                        sd_main_duration_percentage = 0
                                        sd_token_per_second = 0
                                        sd_iterations = 0
                                        sd_avg_acceptance_rate = 0
                                        sd_accepted_token_cnt = 0
                                        sd_accepted_token_rate = 0
                                        llm_boost = 0
                                        avg_llm_boost = 0
                                        llm_boost_ = 0
                                        avg_llm_boost_ = 0
                                        cb_boost = 0
                                        avg_cb_boost = 0
                                        cb_boost_ = 0
                                        avg_cb_boost_ = 0

                                        try:
                                            print(command_line_llm)
                                            run(command_line_llm, check=True, shell=True)
                                            llm_duration = parse_log()
                                            llm_token_per_sec = llm_duration / generation_len

                                            avg_llm_duration.append(llm_duration)
                                        except:
                                            pass

                                        try:
                                            print(command_line_cb)
                                            run(command_line_cb, check=True, shell=True)
                                            cb_duration = parse_log()
                                            cd_token_per_sec = cb_duration / generation_len

                                            avg_cb_duration.append(cb_duration)
                                        except:
                                            pass

                                        try:
                                            print(command_line_sd)
                                            run(command_line_sd, check=True, shell=True)
                                            sd_duration, sd_draft_duration, sd_main_duration, sd_draft_duration_percentage, \
                                                sd_main_duration_percentage, sd_token_per_second, sd_iterations, sd_avg_acceptance_rate, sd_accepted_token_cnt, sd_accepted_token_rate = sd_parse_log()
                                            avg_sd_duration.append(sd_duration)
                                            avg_sd_draft_duration.append(sd_draft_duration)
                                            avg_sd_main_duration.append(sd_main_duration)
                                            avg_sd_draft_duration_percentage.append(sd_draft_duration_percentage)
                                            avg_sd_main_duration_percentage.append(sd_main_duration_percentage)
                                            avg_sd_iterations += sd_iterations
                                            avg_sd_avg_acceptance_rate += sd_avg_acceptance_rate
                                            avg_sd_accepted_token_cnt += sd_accepted_token_cnt
                                            avg_sd_accepted_token_rate += sd_accepted_token_rate
                                        except:
                                            pass

                                        llm_boost = llm_duration / sd_duration
                                        llm_boost_ = llm_boost * 100 - 100
                                        cb_boost = cb_duration / sd_duration
                                        cb_boost_ = cb_boost * 100 - 100

                                        csv_writer_all.writerow([
                                            model_pair[0], main_element_type, model_pair[1], draft_element_type,
                                            sd_type, sampling, generation_len, devices[0], devices[0],
                                            llm_token_per_sec, cd_token_per_sec, sd_token_per_second,
                                            sd_iterations, sd_accepted_token_cnt, sd_accepted_token_rate, sd_avg_acceptance_rate,
                                            sd_draft_duration, sd_draft_duration_percentage, sd_main_duration, sd_main_duration_percentage,
                                            llm_duration, cb_duration, sd_duration,
                                            llm_boost, llm_boost_, cb_boost, cb_boost_
                                        ])

                                        if run_id == len(num_run) - 1:
                                            l = len(num_run)
                                            avg_cd_token_per_sec /= l
                                            avg_sd_token_per_second /= l
                                            avg_sd_iterations /= l
                                            avg_sd_accepted_token_cnt /= l
                                            avg_sd_accepted_token_rate /= l
                                            avg_sd_avg_acceptance_rate /= l

                                            avg_llm_duration = median(avg_llm_duration)
                                            avg_cb_duration = median(avg_cb_duration)
                                            avg_sd_duration = median(avg_sd_duration)
                                            avg_sd_draft_duration = median(avg_sd_draft_duration)
                                            avg_sd_main_duration = median(avg_sd_main_duration)
                                            avg_sd_draft_duration_percentage = median(avg_sd_draft_duration_percentage)
                                            avg_sd_main_duration_percentage = median(avg_sd_main_duration_percentage)
                                            
                                            avg_llm_token_per_sec = avg_llm_duration / generation_len
                                            avg_cb_token_per_sec = avg_cb_duration / generation_len
                                            avg_sd_token_per_second = avg_sd_duration / generation_len

                                            avg_llm_boost = avg_llm_duration / avg_sd_duration
                                            avg_cb_boost = avg_cb_duration / avg_sd_duration
                                            avg_llm_boost_ = avg_llm_boost * 100 - 100
                                            avg_cb_boost_ = avg_cb_boost * 100 - 100


                                    csv_writer.writerow([
                                            model_pair[0], main_element_type, model_pair[1], draft_element_type,
                                            sd_type, sampling, generation_len, devices[0], devices[0],
                                            avg_llm_token_per_sec, avg_cb_token_per_sec, avg_sd_token_per_second,
                                            avg_sd_iterations, avg_sd_accepted_token_cnt, avg_sd_accepted_token_rate, avg_sd_avg_acceptance_rate,
                                            avg_sd_draft_duration, avg_sd_draft_duration_percentage, avg_sd_main_duration, avg_sd_main_duration_percentage,
                                            avg_llm_duration, avg_cb_duration, avg_sd_duration,
                                            avg_llm_boost, avg_llm_boost_, avg_cb_boost, avg_cb_boost_
                                        ])


