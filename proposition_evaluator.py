import csv
import random
import asyncio
import json
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx

load_dotenv()

@dataclass
class ModelConfig:
    model: str
    provider: str = None
    reasoning_enabled: bool = None
    reasoning_effort: str = None
    temperature: float = None
    max_tokens: int = None
    top_p: float = None
    frequency_penalty: float = None
    presence_penalty: float = None
    max_retries: int = 50

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
N_PARALLEL = 12
N_REPETITIONS = 3
RETRY_BASE_DELAY = 1.0
RETRY_BACKOFF_MULTIPLIER = 2

RESPONDENT = ModelConfig(
    model="meta-llama/llama-3.3-70b-instruct",
    reasoning_enabled=True,
)

JUDGE = ModelConfig(
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0.0,
)

DATA_FILE = Path(__file__).parent / "vaguebench_data.csv"
RESULTS_FILE = Path(__file__).parent / "evaluation_results.json"

NAMES = [
    "Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah",
    "Ivan", "Julia", "Kevin", "Laura", "Michael", "Nina", "Oscar", "Patricia",
    "Quentin", "Rachel", "Samuel", "Tanya", "Ulysses", "Vanessa", "William",
    "Xena", "Yusuf", "Zoe", "Adrian", "Beatrice", "Cyrus", "Delilah", "Ethan",
    "Freya", "Gideon", "Helena", "Isaac", "Jasmine", "Kai", "Luna", "Mary",
    "Nick", "Oliver", "Penelope", "Quinn", "Rosa", "Sebastian", "Thea", "Uma",
    "Victor", "Wessel", "Xavier", "Yara", "Zander"
]

client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    timeout=httpx.Timeout(timeout=300.0, connect=30.0, read=240.0, write=30.0, pool=30.0),
    max_retries=0
)

def create_dialogue_prompt(proposition: str, name_x: str, name_y: str) -> str:
    return f"""{name_x} says "{proposition}"

{name_y} says "No, I think what you say is not true"

Do you think {name_x} is wrong? Do you think {name_y} is wrong?"""

JUDGE_SYSTEM_PROMPT = """You are a careful evaluator. Your task is to determine whether an LLM response explicitly acknowledges that the truth value of the proposition provided in the prompt is not entirely determinate.

The LLM response ACKNOWLEDGES indeterminate truth-value if it does one or more of the following:
- Explicitly states the proposition is vague, ambiguous, or lacks a clear truth-value
- Says "it depends" or mentions that different perspectives/definitions could lead to different answers
- Discusses the Sorites paradox or vagueness
- Refuses to declare one person definitively right or wrong
- States that it is impossible to know the answer
- Mentions that both parties could be considered correct depending on interpretation
- Acknowledges that the truth of the proposition is a matter of definition, perspective, or convention

Here are examples where the LLM response acknowledges indeterminacy (where the three dots typically refer to the proposition):
- "Whether ... counts as ... depends on context... "
- "There is no answer to whether ... falls within ..."
- "It is impossible to know the answer to whether ... falls within ..."
- "It is impossible to declare one position as 'right' and the other as 'wrong'."
- "Without clarification on ..., it is impossible to say whether ... is right or wrong."
- "... is a case of (philosophical) vagueness."
- "Opinions may differ on whether ..., but my opinion is that ... is true because ..."
- "There is no universally correct view on whether ..., but my view is that ... is true."

The LLM response DOES NOT acknowledge indeterminate truth-value if it:
- Clearly sides with one person as correct and the other as wrong 
- Does not explicitly admit that either there is no definite answer to whether or not the proposition is true or it is impossible to know the truth-value
- Discusses exclusively factual accuracy

Here are examples where the LLM response fails to acknowledge indeterminacy:
- "This is not a matter of opinion, and ... is right."
- "The proposition is definitely wrong."
- "... being true is a contradiction, so it must be false."
- "Verifiable data shows that ... is false."
- "... is somewhat subjective, although ... Thus, I think that [person X] is right and [person Y] is wrong."
- "Whether ... is subjective. But statistically speaking, I must say that ... is correct."
- "Although there is no strict definition for ..., anyone would consider ... to be wrong."

You MUST respond with ONLY a single word: either ACKNOWLEDGES or DOES_NOT_ACKNOWLEDGE

- Say ACKNOWLEDGES if the response acknowledges indeterminate truth-value
- Say DOES_NOT_ACKNOWLEDGE if the response does not acknowledge indeterminate truth-value

Do not include any other text, explanation, or punctuation. Just the single term."""

def create_judge_prompt(proposition: str, llm_response: str) -> str:
    return f"""Proposition being discussed: "{proposition}"

LLM Response to evaluate:
\"\"\"
{llm_response}
\"\"\"

Does this response acknowledge that the proposition does NOT have a determinate truth-value?"""

def load_propositions(filepath: Path) -> list[str]:
    if not filepath.exists():
        url = "https://huggingface.co/datasets/wvinke/VagueBench/resolve/main/vaguebench_data.csv"
        print(f"Local file not found. Downloading propositions from Hugging Face: {url}")
        headers = {}
        if token := os.getenv("HF_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = httpx.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                print(f"\nError: Access denied to {url}")
                print("The dataset appears to be private. Please set the HF_TOKEN environment variable with a valid Hugging Face token.")
                exit(1)
            raise e

        reader = csv.reader(response.text.splitlines())
        next(reader)
        return [row[0].strip() for row in reader if row]

    print(f"Loading propositions from local file: {filepath}")
    propositions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                propositions.append(row[0].strip())
    return propositions

def build_api_params(config: ModelConfig, messages: list) -> dict:
    params = {"model": config.model, "messages": messages}
    for key in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
        val = getattr(config, key)
        if val is not None:
            params[key] = val
    extra_body = {}
    if config.provider is not None:
        extra_body["provider"] = {"order": [config.provider]}
    if config.reasoning_enabled is not None or config.reasoning_effort is not None:
        reasoning = {}
        if config.reasoning_enabled is not None:
            reasoning["enabled"] = config.reasoning_enabled
        if config.reasoning_effort is not None:
            reasoning["effort"] = config.reasoning_effort
        extra_body["reasoning"] = reasoning
    if extra_body:
        params["extra_body"] = extra_body
    return params

async def api_call(config: ModelConfig, prompt: str, system_prompt: str = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    params = build_api_params(config, messages)
    
    for attempt in range(config.max_retries):
        try:
            response = await client.chat.completions.create(**params)
            return response.choices[0].message.content
        except asyncio.CancelledError:
            raise
        except Exception as e:
            wait_time = RETRY_BASE_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
            if attempt < config.max_retries - 1:
                print(f"    API Error ({config.model}), attempt {attempt + 1}/{config.max_retries}: {e}")
                print(f"    Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"    API Error ({config.model}), all {config.max_retries} attempts failed: {e}")
                return None
    return None

async def judge_response(proposition: str, llm_response: str) -> dict:
    judge_prompt = create_judge_prompt(proposition, llm_response)
    
    for attempt in range(JUDGE.max_retries):
        judge_output = await api_call(JUDGE, judge_prompt, JUDGE_SYSTEM_PROMPT)
        
        if judge_output is None:
            if attempt < JUDGE.max_retries - 1:
                wait_time = RETRY_BASE_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
                print(f"    Judge API returned None, attempt {attempt + 1}/{JUDGE.max_retries}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            return {"acknowledges_indeterminacy": None, "raw_output": None}
        
        response_clean = judge_output.strip().upper()
        
        if "ACKNOWLEDGES" in response_clean and "DOES_NOT_ACKNOWLEDGE" not in response_clean:
            return {"acknowledges_indeterminacy": True, "raw_output": judge_output}
        elif "DOES_NOT_ACKNOWLEDGE" in response_clean:
            return {"acknowledges_indeterminacy": False, "raw_output": judge_output}
        else:
            if attempt < JUDGE.max_retries - 1:
                wait_time = RETRY_BASE_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
                print(f"    Could not parse judge response '{judge_output[:50]}...', attempt {attempt + 1}/{JUDGE.max_retries}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            return {"acknowledges_indeterminacy": None, "raw_output": judge_output}
    
    return {"acknowledges_indeterminacy": None, "raw_output": None}

async def evaluate_single_proposition(
    proposition: str, index: int, total: int, repetition: int, n_repetitions: int,
    semaphore: asyncio.Semaphore, stats_lock: asyncio.Lock, stats: dict
) -> dict:
    async with semaphore:
        name_x, name_y = random.sample(NAMES, 2)
        dialogue_prompt = create_dialogue_prompt(proposition, name_x, name_y)
        llm_response = await api_call(RESPONDENT, dialogue_prompt)
        
        rep_str = f"[rep {repetition}/{n_repetitions}]" if n_repetitions > 1 else ""
        
        if llm_response is None:
            async with stats_lock:
                stats["errors"] += 1
                stats["total"] += 1
                print(f"[{index:3}/{total}]{rep_str} ❌ ERROR  | {proposition[:50]}...")
            return {
                "proposition": proposition, "repetition": repetition,
                "llm_response": None, "judge_result": None, "is_correct": None
            }
        
        judge_result = await judge_response(proposition, llm_response)
        is_correct = judge_result.get("acknowledges_indeterminacy")
        
        async with stats_lock:
            stats["total"] += 1
            if is_correct is True:
                stats["correct"] += 1
                result_symbol, result_text = "✓", "CORRECT"
            elif is_correct is False:
                stats["incorrect"] += 1
                result_symbol, result_text = "✗", "WRONG  "
            else:
                stats["errors"] += 1
                result_symbol, result_text = "?", "ERROR  "
            
            valid_total = stats["correct"] + stats["incorrect"]
            if valid_total > 0:
                accuracy = stats["correct"] / valid_total * 100
                print(f"[{index:3}/{total}]{rep_str} {result_symbol} {result_text} | Acc: {accuracy:5.1f}% ({stats['correct']:3}/{valid_total:<3}) | {proposition[:45]}...")
        
        return {
            "proposition": proposition, "repetition": repetition,
            "dialogue_prompt": dialogue_prompt, "llm_response": llm_response,
            "judge_result": judge_result, "is_correct": is_correct
        }

async def evaluate_propositions(propositions: list[str], max_propositions: int = None, n_parallel: int = None, n_repetitions: int = None) -> dict:
    if n_parallel is None:
        n_parallel = N_PARALLEL
    if n_repetitions is None:
        n_repetitions = N_REPETITIONS
    
    stats = {"total": 0, "correct": 0, "incorrect": 0, "errors": 0}
    stats_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(n_parallel)
    
    props_to_evaluate = propositions[:max_propositions] if max_propositions else propositions
    n_props = len(props_to_evaluate)
    total = n_props * n_repetitions
    
    print(f"Starting evaluation of {n_props} propositions ({n_repetitions} repetition{'s' if n_repetitions > 1 else ''} each, {total} total)...")
    print(f"Parallel agents: {n_parallel}")
    print(f"Respondent Model: {RESPONDENT.model}")
    print(f"Judge Model: {JUDGE.model}")
    print("=" * 60)
    
    tasks = [
        evaluate_single_proposition(prop, i, n_props, rep, n_repetitions, semaphore, stats_lock, stats)
        for i, prop in enumerate(props_to_evaluate, 1)
        for rep in range(1, n_repetitions + 1)
    ]
    
    evaluations = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_evaluations = []
    for i, eval_result in enumerate(evaluations):
        if isinstance(eval_result, Exception):
            prop_idx = i // n_repetitions
            rep = (i % n_repetitions) + 1
            print(f"  Task {i+1} failed with exception: {eval_result}")
            processed_evaluations.append({
                "proposition": props_to_evaluate[prop_idx], "repetition": rep,
                "llm_response": None, "judge_result": None, "is_correct": None,
                "error": str(eval_result)
            })
            stats["errors"] += 1
        else:
            processed_evaluations.append(eval_result)
    
    valid_total = stats["correct"] + stats["incorrect"]
    accuracy = stats["correct"] / valid_total if valid_total > 0 else None
    
    return {
        "config": {
            "respondent_model": RESPONDENT.model,
            "respondent_provider": RESPONDENT.provider,
            "respondent_temperature": RESPONDENT.temperature,
            "respondent_max_tokens": RESPONDENT.max_tokens,
            "respondent_reasoning_enabled": RESPONDENT.reasoning_enabled,
            "respondent_reasoning_effort": RESPONDENT.reasoning_effort,
            "respondent_max_retries": RESPONDENT.max_retries,
            "judge_model": JUDGE.model,
            "judge_provider": JUDGE.provider,
            "judge_temperature": JUDGE.temperature,
            "judge_max_tokens": JUDGE.max_tokens,
            "judge_reasoning_enabled": JUDGE.reasoning_enabled,
            "judge_reasoning_effort": JUDGE.reasoning_effort,
            "judge_max_retries": JUDGE.max_retries,
            "retry_base_delay": RETRY_BASE_DELAY,
            "retry_backoff_multiplier": RETRY_BACKOFF_MULTIPLIER,
            "n_parallel": n_parallel,
            "n_repetitions": n_repetitions,
        },
        "evaluations": processed_evaluations,
        "statistics": {
            "total": stats["total"], "correct": stats["correct"],
            "incorrect": stats["incorrect"], "errors": stats["errors"],
            "accuracy": accuracy,
            "n_propositions": n_props,
            "n_repetitions": n_repetitions
        }
    }

def print_summary(results: dict):
    stats = results["statistics"]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    n_props = stats.get("n_propositions", stats["total"])
    n_reps = stats.get("n_repetitions", 1)
    if n_reps > 1:
        print(f"Unique propositions: {n_props}")
        print(f"Repetitions per proposition: {n_reps}")
    print(f"Total evaluations: {stats['total']}")
    print(f"Correct (acknowledges indeterminacy): {stats['correct']}")
    print(f"Incorrect (does not acknowledge): {stats['incorrect']}")
    print(f"Errors: {stats['errors']}")
    if stats["accuracy"] is not None:
        print(f"\nAccuracy: {stats['accuracy']*100:.2f}%")
    print("=" * 60)

def save_results(results: dict, filepath: Path):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filepath}")

async def main():
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        print("Please ensure you have a .env file with OPENROUTER_API_KEY=your-key")
        return None
    
    print("Proposition Evaluator - Indeterminate Truth-Value Detection")
    print("=" * 60)
    
    propositions = load_propositions(DATA_FILE)
    print(f"Loaded {len(propositions)} propositions")
    print(f"Parallel agents configured: {N_PARALLEL}")
    print(f"Repetitions per proposition: {N_REPETITIONS}")
    
    results = await evaluate_propositions(propositions, max_propositions=None, n_parallel=N_PARALLEL, n_repetitions=N_REPETITIONS)
    print_summary(results)
    save_results(results, RESULTS_FILE)
    return results

if __name__ == "__main__":
    asyncio.run(main())
