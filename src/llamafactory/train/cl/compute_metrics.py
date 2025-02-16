import string
import json
import os
import argparse
import logging
from datasets import load_metric
from .rouge import rouge_scorer
from transformers import AutoTokenizer
from fuzzywuzzy import fuzz
import re
import copy


logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)
GPT2TOKENIZER = os.path.join(CURRENT_DIR, "../data/gpt2tokenizer")


class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word. 
        # But for the first word of a sentence, there is no space before it. 
        # So, we remove all the added spaces ("Ġ"). 
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens


xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def sari_score(inputs, predictions, ground_truths):
    sari = load_metric("src/llamafactory/train/cl/sari")
    translation_result = sari.compute(sources=inputs, predictions=predictions, references=[[label] for label in ground_truths])
    return translation_result


def postprocess(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code


def caculate_fuzz(results, data):
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id] 
        if prediction == "" or target == "":
            continue
        scores += fuzz.ratio(prediction, target)
    avg_score = scores / len(results) 
    return avg_score


def sim_score(predictions, ground_truths):
    outputs = []
    for output in predictions:
        outputs.append(postprocess(output))
    gts = []
    for gt in ground_truths:
        gts.append(postprocess(gt))

    fuzz = caculate_fuzz(outputs, gts)
    evaluation_result = {"similarity": fuzz}
    
    return evaluation_result


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    try:
        return max(scores_for_ground_truths)
    except:
        return 0


def compute_metrics(predictions, references, xlingual=False, inputs=None, group=None, categories=None):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, rouge1, rougeL = 0, 0, 0
    for pred, gold in zip(predictions, references):
        if group == 'scienceqa':
            ori_pred, ori_gold = copy.deepcopy(pred), copy.deepcopy(gold)
            pred = ori_pred[0]
            gold = ori_gold[0]
        gold = [gold]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        if group == 'scienceqa':
            pred = ori_pred[2:]
            gold = ori_gold[2:]
            gold = [gold]
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL}
    # metrics = {"exact_match": exact_match, "rouge1": rouge1, "rougeL": rougeL, "sari": sari, "similarity": sim}
    if group == '20minuten':
        sari = sari_score(predictions=predictions, ground_truths=references, inputs=inputs)['sari']
        metrics.update({"sari": sari})
    if group == 'py150':
        sim = sim_score(predictions=predictions, ground_truths=references)['similarity']
        metrics.update({"similarity": sim})
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False, inputs=None):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group, input in zip(predictions, references, groups, inputs):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold, input))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references, inputs = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual, inputs=inputs, group=group)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.predictions) as fin:
        examples = [json.loads(l) for l in fin]

    predictions = [e["prediction"] for e in examples]
    references = [e["instance"]["output"] for e in examples]
    tasks = []
    for e in examples:
        if e["task"] == "task121_atomic_question_rewriting":
            e["task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    results = compute_metrics(predictions, references, xlingual=args.track == "xlingual")
    print("======== Overall Metrics ========")
    print("all_rougeL", results["rougeL"])
    print("all_EM", results["exact_match"])
    print()
    
    category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
    ]
    category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}

    if args.compute_per_category_metrics:
        print("======== Metrics per category ========")
        task_category = {}
        for task in set(tasks):
            with open(os.path.join("./data/tasks/", task+".json")) as fin:
                task_data = json.load(fin)
                task_category[task] = "_".join(task_data["Categories"][0].lower().split())
        categories = [task_category[e["Task"]] for e in examples] 
        results.update(compute_grouped_metrics(predictions, references, categories, xlingual=args.track=="xlingual"))
        
        for category, metric in category_metrics.items():
            # category = "_".join(category.lower().split())
            if f"{metric}_for_{category}" in results:
                print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
        print()
            
    if args.compute_per_task_metrics:
        print("======== Metrics per task ========")
        results_by_task = compute_grouped_metrics(predictions, references, tasks, xlingual=args.track=="xlingual")
        for task in sorted(list(set(tasks))):
            category = task_category[task]
            metric = category_metrics[category]
            print(task, results_by_task[f"{metric}_for_{task}"])
        print()