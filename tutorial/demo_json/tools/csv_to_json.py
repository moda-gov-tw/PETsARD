import csv
import json
import sys
import os

def is_empty(value):
    """檢查值是否為空（支援 None、空字串、全空白字串）"""
    if value is None:
        return True
    return str(value).strip() == ''

def clean_csv_to_json(csv_filename, json_filename):
    # 檢查輸入檔案是否存在
    if not os.path.isfile(csv_filename):
        print(f"錯誤：輸入檔案 '{csv_filename}' 不存在。")
        sys.exit(1)

    try:
        with open(csv_filename, mode='r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                print("錯誤：CSV 檔案無欄位標頭或為空。")
                sys.exit(1)

            cleaned_data = []
            total_rows = 0
            dropped_rows = 0

            for row in reader:
                total_rows += 1
                # 檢查整列是否全為空（跳過）
                # 任一欄位有空值 → 跳過該列
                skip = False
                for key, value in row.items():
                    if is_empty(value):
                        skip = True
                        break
                if skip:
                    dropped_rows += 1
                    continue

                # 轉換型態（可選）：將數字字串轉為 int/float
                cleaned_row = {}
                for key, value in row.items():
                    v = value.strip()
                    if v == '':
                        # 重複檢查（理論上不會觸發，因已過濾）
                        skip = True
                        break
                    try:
                        if '.' in v:
                            cleaned_row[key] = float(v)
                        else:
                            cleaned_row[key] = int(v)
                    except ValueError:
                        cleaned_row[key] = v

                if not skip:
                    cleaned_data.append(cleaned_row)

        # 寫入 JSON
        with open(json_filename, mode='w', encoding='utf-8') as json_file:
            json.dump(cleaned_data, json_file, ensure_ascii=False, indent=2)

        print(f"✅ 成功！")
        print(f"   - 原始資料列數：{total_rows}")
        print(f"   - 移除空值後保留列數：{len(cleaned_data)}")
        print(f"   - 跳過列數：{dropped_rows}")
        print(f"   - 輸出至：'{json_filename}'")

    except UnicodeDecodeError:
        # 嘗試 Big5 編碼（常見於台灣 Excel 匯出 CSV）
        try:
            with open(csv_filename, mode='r', encoding='big5', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                cleaned_data = []
                total_rows = 0
                dropped_rows = 0

                for row in reader:
                    total_rows += 1
                    skip = False
                    for key, value in row.items():
                        if is_empty(value):
                            skip = True
                            break
                    if skip:
                        dropped_rows += 1
                        continue

                    cleaned_row = {}
                    for key, value in row.items():
                        v = value.strip()
                        try:
                            if '.' in v:
                                cleaned_row[key] = float(v)
                            else:
                                cleaned_row[key] = int(v)
                        except ValueError:
                            cleaned_row[key] = v
                    cleaned_data.append(cleaned_row)

            with open(json_filename, mode='w', encoding='utf-8') as json_file:
                json.dump(cleaned_data, json_file, ensure_ascii=False, indent=2)

            print(f"✅ 成功（使用 Big5 編碼）！")
            print(f"   - 原始資料列數：{total_rows}")
            print(f"   - 移除空值後保留列數：{len(cleaned_data)}")
            print(f"   - 跳過列數：{dropped_rows}")
            print(f"   - 輸出至：'{json_filename}'")

        except Exception as e2:
            print(f"錯誤：處理失敗 - {e2}")
            sys.exit(1)

    except Exception as e:
        print(f"錯誤：處理失敗 - {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python clean_csv_to_json.py <輸入.csv> <輸出.json>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_json = sys.argv[2]

    clean_csv_to_json(input_csv, output_json)
