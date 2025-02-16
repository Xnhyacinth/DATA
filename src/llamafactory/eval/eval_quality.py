import json
import random
import requests
import multiprocessing
from tqdm import tqdm
import re
import time
import openai
from openai import OpenAI
import collections
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def predict(params):
    item, prompt_template, dims, engine = params
    prompt = prompt_template.replace('$INST$', item['prompt']).replace('$RESPONSE$', item["generation"])
    retry_count = 5
    retry_interval = 5
    scores = None
    trys = 0
    if 'yi' in engine:
        API_KEY = "fdf536f26ac9460496ca71695bf58cec"
        API_BASE = "https://api.lingyiwanwu.com/v1"
    elif 'gpt' in engine:
        API_KEY = "sk-GX5fQitXHKizUe4iF8Ed3375A72847A8807c9dAb0290C1Bc"
        # openai.base_url = url
        API_BASE = 'https://chatapi.onechats.top/v1/'
    elif 'glm' in engine:
        API_KEY = "a34d5b2c2d93599b50975262f17c2557.9zsLgHWZXmPzrSF0"
        # openai.base_url = url
        API_BASE = 'https://open.bigmodel.cn/api/paas/v4/'
    else:
        API_KEY = "sk-c7n62ac5h44ynsrk"
        API_BASE = f'https://cloud.infini-ai.com/maas/{engine}/nvidia/'
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE
    )

    while scores is None and trys < retry_count:
        try:
            response = client.chat.completions.create(
                model=engine,  # use gpt4
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating text quality. Please evaluate the quality of an AI assistant's response to a user's writing request. Be as strict as possible."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                # stream=True
            )
            output = response.choices[0].message.content.strip()
            # if '```json' in output:
            #     output = extract_info(r'```json\n(.*?)\n```', output)
            # output = output.replace('\n', '')
            scores = json.loads(output)
            for dim in dims:
                if dim not in scores:
                    scores = None
                    trys += 1
        except TimeoutError:
            print("Timeout", prompt)
            trys += 1
            # retry_interval *= 2
            time.sleep(retry_interval)

        except Exception as e:
            print(e)
            # print(question)
            trys += 1
            # retry_interval *= 2
            time.sleep(retry_interval)
    # import pdb;pdb.set_trace()
    if scores is None:
        print(output)
        item['quality_scores'] = output
    else:
        item['quality_scores'] = scores
    return item

def score_quality(data, prompt_file='judge.txt', engine='gpt-4o-mini'):
    dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience", "Completeness", "Engagement", "Consistency", "Structure"]
    prompt_template = open(prompt_file, "r", encoding="utf-8").read()

    pbar = tqdm(total=len(data))
    index = 0
    pbar.update(index)
    # predict((data[0], prompt_template, dims, engine))
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(predict, (item, prompt_template, dims, engine)) for item in data]
        query2res = collections.defaultdict(int)
        results = []

        for job in as_completed(futures):
            res = job.result(timeout=None)
            # res['gen_length'] = count_words(res['generation'])
            # res['response'] = res['summary']
            # del res['summary']
            results.append(res)
                
            # query2res[query] = res
            pbar.update(1)
    # results = []
    # for k, v in query2res.items():
    #     results.append(v)
    # with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
    all_scores = [line['quality_scores'] for line in results]

    total_score = dict()
    for dim in dims:
        scores = [float(score[dim]) if dim in score else 6 for score in all_scores]
        # total_score[dim] = ((sum(scores) / len(scores)) - 1) * 25
        total_score[dim] = sum(scores) / len(scores) * 10
    total_score['total'] = sum(total_score.values()) / len(total_score)
    print(total_score)
    # import pdb;pdb.set_trace()
    pbar.close()
    
    return results, total_score

def eval_quality(model, filename, prompt_file='judge.txt', engine='gpt-4o-mini'):
    dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience", "Completeness", "Engagement", "Consistency", "Structure"]
    filename = f"{model}/{filename}"
    prompt_template = open(prompt_file, "r", encoding="utf-8").read()
    outfile = f'eval/{model}/{filename.split("/")[1]}'
    os.makedirs(outfile, exist_ok=True)
    # outs = open(out_name, 'w', encoding='utf-8')
    outfile += '/quality.json'
    
    # if not os.path.exists(outfile):
    outs = open(outfile, 'w', encoding='utf8')
# GPT4_API_KEY = '' # Your API Key
# GPT_MODEL = 'gpt-4o-mini'
    with open(filename, 'r', encoding='utf-8') as filename:
        data = json.load(filename)
    random.shuffle(data)
    
    pbar = tqdm(total=len(data))
    index = 0
    pbar.update(index)
    # predict((data[0], prompt_template, dims, engine))
    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(predict, (item, prompt_template, dims, engine)) for item in data]
        query2res = collections.defaultdict(int)
        results = []

        for job in as_completed(futures):
            res = job.result(timeout=None)
            # res['gen_length'] = count_words(res['generation'])
            # res['response'] = res['summary']
            # del res['summary']
            results.append(res)
                
            # query2res[query] = res
            pbar.update(1)
    # results = []
    # for k, v in query2res.items():
    #     results.append(v)
    # with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
    all_scores = [line['quality_scores'] for line in results]

    total_score = dict()
    for dim in dims:
        scores = [float(score[dim]) if dim in score else 6 for score in all_scores]
        # total_score[dim] = ((sum(scores) / len(scores)) - 1) * 25
        total_score[dim] = sum(scores) / len(scores) * 10
    total_score['total'] = sum(total_score.values()) / len(total_score)
    print(total_score)
    
    json.dump(results, outs, indent=4)
    # import pdb;pdb.set_trace()
    pbar.close()
    outs.close()
    
    return results, total_score
    # else:
    #     results = json.load(open(outfile, 'r', encoding='utf-8'))
    #     outs = open(outfile, 'w', encoding='utf8')
        
    # all_scores = [line['quality_scores'] for line in results]

    # total_score = dict()
    # for dim in dims:
    #     scores = [float(score[dim]) if dim in score else 6 for score in all_scores]
    #     # total_score[dim] = ((sum(scores) / len(scores)) - 1) * 25
    #     total_score[dim] = sum(scores) / len(scores) * 10
    # total_score['total'] = sum(total_score.values()) / len(total_score)
    # print(total_score)
    # macro = {
    #     'quality_scores': total_score
    # }
    # outs = open(outfile, 'w', encoding='utf8')
    # json.dump([macro, results], outs, indent=4)
    # outs.close()
        
# eval_quality('gpt-4o-mini', 'gpt-4o-mini/story/story-test-p1_0shot.json')