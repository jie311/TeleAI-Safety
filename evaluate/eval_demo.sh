cd ./evaluate
python eval_summary_report.py \
  --scorer=PatternScorer \
  --config_path=./config/pattern_scorer.yaml \
  --json_path=./results/dra_gpt_4o_mini.jsonl \
  --output_path=./evaluation_report.txt