## Status
Project is still undergoing

0) DONE. 
Set up - folders and deps, parquet files, fill out configs/default.yaml

1 Working on.

WRite core modules and very

- DONE. config_loader: loads configs/default.yaml → Config object
- DONE. data_loader. THIS WAS A PAIN!!
- DONE. prompt_parser
- DONE. HF model wrapper
- DONE. evaluator. This is the runner.py
- DONE. aggregator. This is metrics.py

2 START HERE. Smoke tests???....use Val dataset???? Need to thinknabout. ON run_val_sweep.py file


3 Validation

4 Final test (hold-outscore)

5 Plots

6

## Inspiration for this project

On my road to become an AI person, this project came about for two reasons. The first reason came from my own endless curiosity. Optimization was a topic that I read about on the web and wanted to know more. In my own understanding, optimization can mean many things such as accuracy, latency, I/O, etc., the list goes on and on. One component is internal, onesuch is internal knobs like temperature. Another compoent is hardware and cloud services like an NVIDIA or AMD GPU and Google GPU, respectively. The latter componenent cost money, so I thought to myself that I would try to optimize internally. The second reason is that some job postings preferred to have candidates to have some experience in optimization, and although it was NOT a hard requirement, I took the task upon myself to learn something new, out of my own free will, and gain a new perspective and appreciation in the growing, meteoric field of AI.

## Introduction
LLMs can answer multiple‑choice questions reasonably well, but decoding parameters/knobs, especially temperature and  top-p, strongly influence both accuracy and cost. In production, teams tune these knobs to deliver reliable outputs without overspending on tokens. 

This project tunes **inference-time decoding** specifically **temperature** and **top-p** to maximize accuracy while controlling randomness and token cost. We sweep these knobs on a fixed zero-shot prompt across MMLU subjects, select the best setting on the validation split, then freeze it and report final test performance on the test dataset with an emphasis on accuracy and efficiency (accuracy per token).

## Business Case / So What?

The business case is to improve answer accuracy while reducing token spend. It is a an internal config-only change mirroring real world issues in industry. The project answers the quesiton, "Is there a way to optimize how we run the model (decoding settings) while reducing inference costs?" 


## Dataset
The dataset is called the Massive Multitask Learning Understanding (MMLU) dataset that was downloaded from HugginFace. It consist of multi-choice questions of four from multiple fields of educaiton, professional experince, and other branches of knowledge along with correct answers.

I will be using the val and test datasets for this project. The aux_train dataset is not used because it is used for fitting model weights, not for decoding. The the val dataset is a held-out subset and will be used to choose the optimal configuration and used on the test dataset. 

The current landscape of the data are afer cleaning and dropping columns:

Columns:
`question`: A question based on the four subjects above.
`choices`: Four possible answer choices given a list
`answer`: Gold answer

Dimensions:
`val` - 1531 rows
`test` - 14042 rows

## Model

I'm using model `google/gemma-2-2b-it` which is an instruction 2B parameter from HuggingFace. I chose this model because it is lightweight, fast, and local which runs on my Mac MPS. The model itself is alsoer instruction based mearning it is reliable to return a single answer choice with short prompting. I also set one of config settings `max_new_tokens= 1-2` to foce a single letter/digit. 

Meta Product Manager thign on Linkedin By Spencer

## Parameters/Knobs

Need to talk about temperature?

Temperature a knob/hyperparameter that researchers check robustness of the model. Essentialy it represents the degree of creativity or rather hallucinations that the mdo

1) 0.0 → No creativity. Always picks the single most likely answer according to its knowledge.
2) 0.7 → Some creativity. Sometimes picks the 2nd-most likely answer if it’s close in probability.
3) 1.0+ → Lots of creativity. Will occasionally pick even lower-ranked answers.

Need to talk about top-p?
Imagine the model’s answer choices ranked by likelihood: A = 50%, B = 30%, C = 15%, D = 5%.
1) top-p = 1.0 → Consider all possibilities (full list).

2) top-p = 0.9 → Only consider the smallest set of answers whose probabilities add up to 90% (so maybe just A and B in this example).

3) top-p = 0.8 → Even more restrictive — trims the tail.

Need to talk about business case for this project?

Sweeping parameters
“Sweeping” means systematically testing all combinations you care about.

Example:
Temperatures: 0.0, 0.2, 0.5, 0.7
Top-p: 0.8, 0.9, 1.0
That’s 4 × 3 = 12 total setting combinations to try.

For each combination, you:
Run the model on all questions in the dataset.
Record accuracy.
Compare results to find the sweet spot.



In your project, you’re showing you can experiment, measure, and recommend settings.

Need to talk about the grid search?

Why do we care?
The right combination of temperature and top-p can:
Increase accuracy for factual questions (like MMLU Professional Medicine).
Reduce variability between runs (stable results).
Sometimes reduce cost (fewer long-winded answers).

LLM inference cost & performance optimization is a hot skill.
Companies pay for every token generated by an API, and decoding parameters like temperature and top-p directly affect accuracy, creativity, and token count.

If you can tune these knobs to improve accuracy while cutting cost, you deliver measurable business value.

Need to talk about business case for this project?

## Metrics

1) Accuracy = percentage of correct vs. gold
2) Stability = std/variance across K seeds (repeat each setting K times, e.g., K=3)
3) Average tokens = prompt+output tokens per question (proxy for cost)
4) Accuracy per Token = efficiency metric


## Experiment
For each (temperature, top-p) pair:

Run all questions in the Professional Medicine dataset through the model.

Record:

Accuracy (% correct answers)

Stability (if you repeat runs with different seeds)

Tokens used (optional, proxy for cost)

Compare results and find the sweet spot — highest accuracy with good stability.


HF website:
https://huggingface.co/datasets/cais/mmlu/tree/main/professional_medicine

## Procedure

## Results


## Limitations


## Next Steps