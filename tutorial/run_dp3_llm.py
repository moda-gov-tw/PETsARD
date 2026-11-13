# run_dp3_llm.py -- differentially private synthesis 3 of 3: dp-llm-synth (DP_LLM_synthesizer), unstructured text
# 差分隱私合成範例 3/3：dp-llm-synth(DP_LLM_synthesizer)，非結構化資料(文字)
# Needs LM Studio serving openai/gpt-oss-20b on this computer / 需先在本機以 LM Studio 載入 openai/gpt-oss-20b 並開啟 Server
import glob
import json
import sys
import urllib.request

import pandas as pd
import yaml

from petsard import Executor

CONFIG = "demo_dp/configs/dp_llm_bloodpressure.yaml"
REPORT = "demo_dp/output/dp_llm_bloodpressure_output/privacy_report.json"
RESULT = "demo_dp/output/dp_llm_bloodpressure_*.csv"
LOGS = "demo_dp/output/dp_llm_bloodpressure_logs"

# Check LM Studio before anything reads the data, so a run that cannot write text spends no privacy budget
# 先確認 LM Studio 已開啟並提供模型：無法生成文字時不讀取資料，也不花用隱私預算
with open(CONFIG, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["Synthesizer"]["dp_llm"]
try:
    with urllib.request.urlopen(cfg["api_url"].rstrip("/") + "/models", timeout=5) as r:
        models = [m.get("id") for m in json.load(r).get("data", [])]
except Exception:
    sys.exit(f"無法連線到LM Studio({cfg['api_url']})：請先開啟LM Studio、載入模型並啟動Server")
if cfg["model"] not in models:
    sys.exit(f"LM Studio沒有提供模型{cfg['model']}：請在LM Studio載入此模型後再執行")

exec = Executor(config=CONFIG)
exec.run()

with open(REPORT, encoding="utf-8") as f:
    report = json.load(f)
spent = report["epsilon_spent"]
spent = spent if isinstance(spent, str) else f"{spent:.4f}"
print(f"隱私預算：設定ε={report['epsilon_budget']}，實際使用ε={spent}，δ={report['delta']}")

files = sorted(glob.glob(RESULT))
rows = len(pd.read_csv(files[-1]).dropna(how="all")) if files else 0
if rows == 0:
    sys.exit(f"沒有產生文字：請確認LM Studio的模型與Server設定，執行紀錄在{LOGS}")
print(f"執行完成！共產生{rows}筆")
