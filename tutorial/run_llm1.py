# run_llm1.py -- text example of the course recording, run with dp-llm-synth (DP_LLM_synthesizer)
# 課程錄影之文字合成範例：改以更新後方法 dp-llm-synth(DP_LLM_synthesizer)執行，內容與 run_dp3_llm.py 相同
#
# The course recording ran this script with the initial LLM synthesizer; that version is kept in the
# initial-methods branch (tag initial-methods-v1). 課程錄影中本程式使用初版LLM合成器，初版範例保留於 initial-methods 分支。
# Needs LM Studio serving openai/gpt-oss-20b on this computer / 需先在本機以 LM Studio 載入 openai/gpt-oss-20b 並開啟 Server
import runpy

runpy.run_path("run_dp3_llm.py", run_name="__main__")
