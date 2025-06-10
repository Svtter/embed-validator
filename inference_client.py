import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

sys.path.append(".")
sys.path.append("./src")

import config


def get_current_model() -> Optional[Dict[str, Any]]:
  """
  获取当前加载的模型信息
  :return: 包含当前模型和可用模型列表的字典，如果发生错误则返回 None
  """
  url = f"{config.test_host}/model"
  try:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.RequestException as e:
    print(f"获取模型信息时发生错误: {e}")
    return None


def switch_model(model_name: str) -> bool:
  """
  切换当前加载的模型
  :param model_name: 要切换到的模型名称
  :return: 切换是否成功
  """
  url = f"{config.test_host}/model/{model_name}"
  try:
    response = requests.post(url)
    response.raise_for_status()
    return True
  except requests.exceptions.RequestException as e:
    print(f"切换模型时发生错误: {e}")
    if hasattr(e.response, "text"):
      print(f"错误详情: {e.response.text}")
    return False


def perform_inference(input_data: List[float]) -> Optional[Dict[str, Any]]:
  """
  调用推理接口
  :param input_data: 输入数据数组
  :return: 包含推理结果的字典，如果发生错误则返回 None
  """
  url = f"{config.test_host}/inference"
  request_data = {"input_data": input_data}

  try:
    response = requests.post(url, json=request_data)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.RequestException as e:
    print(f"请求发生错误: {e}")
    if hasattr(e.response, "text"):
      print(f"错误详情: {e.response.text}")
    return None


def load_test_data(file_path: str) -> List[float]:
  """
  从 JSON 文件加载测试数据
  :param file_path: JSON 文件路径
  :return: 输入数据数组
  """
  try:
    with open(file_path, "r") as f:
      data = json.load(f)
      return data
  except Exception as e:
    print(f"读取测试数据时发生错误: {e}")
    return []


def perform_multiple_inference(model_name: str, input_data: List[float], num_tests: int = 5) -> List[float]:
  """
  对指定模型执行多次推理测试，忽略第一次推理时间作为预热
  :param model_name: 模型名称
  :param input_data: 输入数据数组
  :param num_tests: 测试次数
  :return: 包含所有测试推理时间的列表（不包含预热结果）
  """
  # 首先切换到指定模型
  if not switch_model(model_name):
    print(f"切换到模型 {model_name} 失败")
    return []

  # 执行一次预热推理
  print("执行预热推理...")
  warmup_result = perform_inference(input_data)
  if not warmup_result:
    print("预热推理失败")
    return []
  time.sleep(1)  # 预热后短暂延迟

  inference_times = []
  for i in range(num_tests):
    print(f"执行第 {i+1}/{num_tests} 次测试...")
    result = perform_inference(input_data)
    if result:
      inference_times.append(result["inference_time"])
    time.sleep(1)  # 添加短暂延迟，避免请求过于频繁
  return inference_times


def main():
  # 获取当前模型信息
  model_info = get_current_model()
  if not model_info:
    print("无法获取模型信息")
    exit(1)

  print("当前加载的模型:", model_info["current_model"])
  print("\n可用的模型列表:")
  available_models = model_info["available_models"]
  for i, model in enumerate(available_models, 1):
    print(f"{i}. {model}")

  # 从文件加载测试数据
  test_data = load_test_data("imgdata.json")
  if not test_data:
    print("无法加载测试数据")
    exit(1)

  print(f"\n输入数据长度: {len(test_data)}")

  # 存储所有模型的测试结果
  all_results = {
    "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "input_data_length": len(test_data),
    "models": {},
  }

  # 测试所有模型
  for model_name in available_models:
    if any(re.match(pattern, model_name) for pattern in config.skip_models):
      print(f"跳过模型: {model_name}")
      continue

    print(f"\n开始测试模型: {model_name}")
    inference_times = perform_multiple_inference(model_name, test_data)

    if inference_times:
      # 计算平均推理时间
      avg_time = sum(inference_times) / len(inference_times)
      all_results["models"][model_name] = {
        "inference_times": inference_times,
        "average_inference_time": avg_time,
        "total_tests": len(inference_times),
      }
      print(f"模型 {model_name} 测试完成")
      print(f"平均推理时间: {avg_time:.4f} 秒")
    else:
      print(f"模型 {model_name} 测试失败")

  # 保存所有测试结果
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  if not os.path.exists("results"):
    os.makedirs("results")

  result_file = f"results/inference_results_{timestamp}.json"
  all_results["test_host"] = config.test_host
  with open(result_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

  print(f"\n所有测试结果已保存到: {result_file}")


if __name__ == "__main__":
  main()
