import logging
import time 
import csv
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO",  # logging level
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
log = logging.getLogger("rich")


import json
import argparse
import pandas as pd
from llmfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from llmfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy,
    LLMMutatorCrossOver,
    LLMMutatorExpand,
    LLMMutatorGenerateSimilar,
    LLMMutatorRephrase,
    LLMMutatorShorten,
)
from llmfuzzer.fuzzer import LLMFuzzer
from llmfuzzer.llms import GPT, Claude, Qwen ,LocalLLM
from llmfuzzer.utils.predict import RoBERTaPredictor


def main(args):
    with open(args.questions_path) as f:
        questions = json.load(f)

    if "gpt" in args.mutator_model.lower():
        mutator_model = GPT(args.mutator_model, args.openai_key, log, args.openai_url)
    elif "gpt" in args.mutator_model.lower():
        mutator_model = Claude(args.mutator_model, args.claude_key, log, args.claude_url)
    else:
        raise NotImplementedError

    if "gpt" in args.target_model.lower():
        target_model = GPT(args.target_model, args.openai_key, log, args.openai_url)
        initial_seed = pd.read_csv("datasets/prompts/GPTFuzzer.csv")["text"].tolist()
    elif "claude" in args.target_model.lower():
        target_model = Claude(args.target_model, args.claude_key, log, args.claude_url)
        initial_seed = pd.read_csv("datasets/prompts/ClaudeFuzzer.csv")["text"].tolist()
    elif "qwen" in args.target_model.lower():
        target_model = Qwen(args.target_model, args.qwen_key, log, args.qwen_url)
        initial_seed = pd.read_csv("datasets/prompts/QwenFuzzer.csv")["text"].tolist()
    elif "llama" in args.target_model.lower():
        target_model = LocalLLM(args.target_model, args.target_model_device, log)
        initial_seed = pd.read_csv("datasets/prompts/LlamaFuzzer.csv")["text"].tolist()
    elif "vicuna" in args.target_model.lower():
        target_model = LocalLLM(args.target_model, args.target_model_device, log)
        initial_seed = pd.read_csv("datasets/prompts/VicunaFuzzer.csv")["text"].tolist()
    else:
        raise NotImplementedError

    if "hubert233/GPTFuzz".lower() in args.predictor_model.lower():
        predictor_model = RoBERTaPredictor(args.predictor_model, device=args.predictor_model_device)
    else:
        raise NotImplementedError

    result_file = args.result_file
    if result_file is None:
        model_name = args.target_model.split("/")[-1]
        result_file = f'results-{model_name}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'
    fp = open(result_file, 'w', buffering=1)
    writter = csv.writer(fp)
    writter.writerow([
        'question_index', 'question', 'is_succeeded', 'num_queries', 'succeeded_prompt', 'succeeded_response'
    ])

    try:
        for idx, question in enumerate(questions):
            log.info(f"Start fuzzing at question: {idx+1}/{len(questions)} |{question}|")
            fuzzer = LLMFuzzer(
                question=question,
                target_model=target_model,
                predictor=predictor_model,
                initial_seed=initial_seed,
                mutate_policy=MutateRandomSinglePolicy(
                    [
                        LLMMutatorCrossOver(mutator_model, 2.0),
                        LLMMutatorExpand(mutator_model, 2.0),
                        LLMMutatorGenerateSimilar(mutator_model, 2.0),
                        LLMMutatorRephrase(mutator_model, 2.0),
                        LLMMutatorShorten(mutator_model, 2.0),
                    ],
                    concatentate=True,
                ),
                select_policy=MCTSExploreSelectPolicy(),
                energy=args.energy,
                max_jailbreak=args.max_jailbreak,
                max_query=args.max_query,
                logger=log,
            )
            result = fuzzer.run()
            writter.writerow([
                idx,
                question,
                result['is_succeeded'],
                result['num_queries'],
                result['succeeded_prompt'],
                result['succeeded_response'],
            ])
    except KeyboardInterrupt:
        log.info("Fuzzing interrupted by user!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuzzing parameters")
    parser.add_argument("--openai_key", type=str, default="", help="OpenAI API Key")
    parser.add_argument("--openai_url", type=str, default="", help="OpenAI url")
    parser.add_argument("--claude_key", type=str, default="", help="Claude API Key")
    parser.add_argument("--claude_url", type=str, default="", help="Claude url")
    parser.add_argument("--qwen_key", type=str, default="", help="Qwen API Key")
    parser.add_argument("--qwen_url", type=str, default="", help="Qwen url")
    parser.add_argument("--questions_path", type=str, default="datasets/questions/behaviors.json", help="The path of malicious questions",)
    parser.add_argument("--mutator_model", type=str, default="gpt-4", help="The mutator model")
    parser.add_argument("--target_model", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="The target model, openai model or open-sourced LLMs",)
    parser.add_argument("--target_model_device", type=str, default="cuda:0", help="device of the local target model",)
    parser.add_argument("--max_query", type=int, default=1000, help="The maximum number of queries")
    parser.add_argument("--max_jailbreak", type=int, default=1, help="The maximum jailbreak number")
    parser.add_argument("--energy", type=int, default=1, help="The energy of the fuzzing process")
    parser.add_argument("--seed_selection_strategy", type=str, default="round_robin", help="The seed selection strategy",)
    parser.add_argument("--max-new-tokens", type=int, default=512, help="The amount of maximum tokens to generate.",)
    parser.add_argument("--predictor_model", type=str, default="hubert233/GPTFuzz", help="The judger model to predict the malicious classes of the responses from LLMs")
    parser.add_argument("--predictor_model_device", type=str, default="cuda:0", help="Device of the local predicotr model",)
    parser.add_argument("--result_file", type=str, default=None, help="The path to save the result csv",)

    args = parser.parse_args()
    log.info(f"args: {args}")
    main(args)