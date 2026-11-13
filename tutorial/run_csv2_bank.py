from petsard import Executor

yaml_path = "demo_csv/configs/config_bank_marketing.yaml"

exec = Executor(config=yaml_path)
exec.run()

print("執行完成！")