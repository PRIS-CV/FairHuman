import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import hpsv2
import ImageReward as RM


model = RM.load(name="/nvfile-heatstorage/zangxh/intern/wangyx/evaluation/ImageReward/ImageReward.pt",med_config="/nvfile-heatstorage/zangxh/intern/wangyx/evaluation/ImageReward/med_config.json")

prompt_list = open("/nvfile-heatstorage/zangxh/intern/wangyx/prompt_test.txt", "r").readlines()
prompts = [prompt.strip() for prompt in prompt_list]

IR_scores =[]
for i, prompt in enumerate(prompts): 
        img_paths = []
        for img_path in os.listdir("/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)):
                img_paths.append("/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)+"/"+img_path)
        # img_paths.append("/home/telenlp/MoLE-master/sd-scripts-xl/gen_command/gen_output/moe_sdxl_gen-id0-llava/"+filename)
        rewards = model.score(prompt, img_paths)
        mean_rewards = sum(rewards)/len(rewards)
        IR_scores.append(mean_rewards)
# import pdb;pdb.set_trace()
print("IR score:", sum(IR_scores)/len(IR_scores))
 