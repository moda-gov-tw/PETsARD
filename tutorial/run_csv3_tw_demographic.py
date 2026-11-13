from petsard import Executor

yaml_path = "demo_csv/configs/config_tw_demographic.yaml"

exec = Executor(config=yaml_path)
exec.run()

print("執行完成！")
