# run_dp1_copula.py -- differentially private synthesis 1 of 3: DP Copula (DPflow_Copula), structured data (CSV)
# 差分隱私合成範例 1/3：DP Copula(DPflow_Copula)，結構化資料(CSV)
import json

from petsard import Executor

exec = Executor(config="demo_dp/configs/dp_copula_adult_income.yaml")
exec.run()

with open("demo_dp/output/dp_copula_adult_income_privacy_report.json", encoding="utf-8") as f:
    report = json.load(f)
print(f"隱私預算：設定ε={report['target_epsilon']}，實際使用ε={report['achieved_epsilon']:.4f}，δ={report['delta']}")
print("執行完成！")
