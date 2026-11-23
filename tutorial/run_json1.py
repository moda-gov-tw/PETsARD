# run_json1.py -- JSON demo 1 of 3: updated DP AdvGAN (DPflow_AdvGAN) on JSON records, with JSON output
# JSON 範例 1/3：以更新版DP AdvGAN(DPflow_AdvGAN)合成JSON資料，合成資料與評估報告以JSON格式輸出
#
# The course recording ran this script with the initial AdvGAN; that version is kept in the
# initial-methods branch (tag initial-methods-v1). 課程錄影中本程式使用初版AdvGAN，初版範例保留於 initial-methods 分支。
import json

from petsard import Executor

exec = Executor(config="demo_json/configs/dp_advgan_json.yaml")
exec.run()

with open("demo_json/output/dp_advgan_json_privacy_report.json", encoding="utf-8") as f:
    report = json.load(f)
print(f"隱私預算：設定ε={report['target_epsilon']}，實際使用ε={report['achieved_epsilon']:.4f}，δ={report['delta']}")
print("執行完成！")
