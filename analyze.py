# 导入需要的库
import numpy as np
import pandas as pd
import statsmodels.api as sm # 用于执行回归分析

# --- Bootstrap 核心函数 ---

def bootstrap_statistic(data, statistic_func, n_bootstrap_samples=1000):
  """
  执行 Bootstrap 重采样，并为每个样本计算指定的统计量。

  参数:
    data: 包含原始数据的 pandas DataFrame。
    statistic_func: 用于计算目标统计量的函数。
    n_bootstrap_samples: Bootstrap 重采样次数 (B)。

  返回:
    一个 numpy 数组，包含所有有效计算出的统计量。
  """
  n_size = data.shape[0]
  bootstrap_stats = []
  print(f"正在生成 {n_bootstrap_samples} 个自助样本并计算统计量...")
  for i in range(n_bootstrap_samples):
    # 1. 有放回地抽取样本索引
    indices = np.random.choice(np.arange(n_size), size=n_size, replace=True)
    bootstrap_sample = data.iloc[indices]

    # 2. 计算该样本的统计量
    try:
      stat = statistic_func(bootstrap_sample)
      if stat is not None and not np.isnan(stat):
          bootstrap_stats.append(stat)
    except Exception as e:
      # 如果计算出错则跳过该样本 (可以选择取消注释下面的打印来看错误)
      # print(f"警告：在第 {i+1} 个自助样本上计算时出错: {e}")
      pass

    # 打印进度
    if (i + 1) % (n_bootstrap_samples // 10) == 0:
        print(f"已完成 {i + 1}/{n_bootstrap_samples}...")

  print("Bootstrap 重采样完成。")
  valid_stats = np.array(bootstrap_stats)
  if len(valid_stats) < n_bootstrap_samples:
      print(f"警告：最终使用了 {len(valid_stats)} 个有效的自助样本结果（总尝试次数：{n_bootstrap_samples}）。")
  return valid_stats


def calculate_confidence_interval(bootstrap_stats, confidence_level=0.95):
  """
  根据 Bootstrap 统计量分布计算置信区间（百分位法）。

  参数:
    bootstrap_stats: bootstrap_statistic 函数返回的统计量数组。
    confidence_level: 置信水平 (如 0.95 代表 95%)。

  返回:
    置信区间的下限和上限 (元组)。
  """
  if len(bootstrap_stats) == 0:
      print("错误：没有有效的自助统计量用于计算置信区间。")
      return (np.nan, np.nan)
  # 计算分位点
  alpha = 1 - confidence_level
  lower_percentile = alpha / 2.0 * 100
  upper_percentile = (1 - alpha / 2.0) * 100
  # 获取置信区间边界
  lower_bound = np.percentile(bootstrap_stats, lower_percentile)
  upper_bound = np.percentile(bootstrap_stats, upper_percentile)
  return lower_bound, upper_bound


def calculate_indirect_effect(data):
  """
  计算简单中介效应 (a*b)，使用 statsmodels.api.OLS。
  模型: 健康 -> 体育活动 -> 客观环境

  参数:
    data: 包含 X, M, Y 列的 pandas DataFrame。

  返回:
    中介效应值 (a*b)，若计算失败则返回 None。
  """
  try:
    # --- 在这里定义 X, M, Y 的列名 (确保与你的数据文件一致) ---
    x_col = '健康'
    m_col = '体育活动'
    y_col = '客观环境'
    # ----------------------------------------------------------

    # 确保列是数值类型，处理 Inf/-Inf 为 NaN
    for col in [x_col, m_col, y_col]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan)

    # --- 估计路径 a (X -> M) ---
    data_a = data[[x_col, m_col]].dropna()
    if data_a.shape[0] < 2: return None
    X_a = sm.add_constant(data_a[x_col])
    y_a = data_a[m_col]
    if X_a.shape[1] > 1 and np.linalg.matrix_rank(X_a) < X_a.shape[1]: return None
    model_a = sm.OLS(y_a, X_a).fit()
    path_a = model_a.params[x_col]

    # --- 估计路径 b (M -> Y | X) ---
    data_b = data[[x_col, m_col, y_col]].dropna()
    if data_b.shape[0] < 3: return None
    X_b = sm.add_constant(data_b[[m_col, x_col]])
    y_b = data_b[y_col]
    if X_b.shape[1] > 1 and np.linalg.matrix_rank(X_b) < X_b.shape[1]: return None
    model_b = sm.OLS(y_b, X_b).fit()
    path_b = model_b.params[m_col]

    # --- 计算中介效应 ---
    indirect_effect = path_a * path_b
    return indirect_effect

  except Exception as e:
    # print(f"计算中介效应时发生错误: {e}") # 出错时可取消注释查看
    return None

# ===== 主程序执行部分 =====
if __name__ == "__main__":

    # --- 1. 设置文件和模型参数 ---
    # 【【【 在这里指定你的 Excel 文件名 】】】
    data_filename = 'data.xlsx'
    # -----------------------------

    # 要计算的统计量函数 (这里是中介效应)
    statistic_to_calculate = calculate_indirect_effect

    # Bootstrap 参数
    num_bootstrap_samples = 5000  # Bootstrap 重采样次数
    confidence_level = 0.95    # 置信水平

    # --- 2. 加载数据 ---
    try:
        all_data = pd.read_excel(data_filename)
        print(f"成功从 '{data_filename}' 加载数据。")
        print("数据前5行预览：")
        print(all_data.head())

        # 检查必需列是否存在 (函数内部会再次检查并使用这些名字)
        required_cols = ['健康', '体育活动', '客观环境']
        if not all(col in all_data.columns for col in required_cols):
            print(f"\n错误：数据文件 '{data_filename}' 中缺少必需的列。")
            print(f"代码需要以下列名: {required_cols}")
            exit() # 缺少列则退出程序

    except FileNotFoundError:
        print(f"\n错误：找不到数据文件 '{data_filename}'。请确保文件名正确且文件在脚本同目录下。")
        exit() # 文件不存在则退出程序
    except Exception as e:
        print(f"\n读取数据文件时发生错误: {e}")
        exit() # 其他读取错误则退出

    # --- 3. 执行 Bootstrap 分析 ---
    print(f"\n开始执行 Bootstrap 分析，计算统计量：{statistic_to_calculate.__name__}")
    bootstrap_distribution = bootstrap_statistic(all_data, statistic_to_calculate, num_bootstrap_samples)
    lower_ci, upper_ci = calculate_confidence_interval(bootstrap_distribution, confidence_level)

    # --- 4. 打印最终结果 ---
    print("\n--- Bootstrap 中介效应结果 ---")
    original_indirect_effect = calculate_indirect_effect(all_data) # 在原始样本上计算一次
    if original_indirect_effect is not None:
        print(f"原始样本计算出的中介效应值 (a*b): {original_indirect_effect:.4f}")
    else:
        print("原始样本计算中介效应失败。")

    print(f"通过 Bootstrap 得到的有效中介效应值数量: {len(bootstrap_distribution)}")
    print(f"{confidence_level*100:.0f}% 置信区间 (百分位法): ({lower_ci:.4f}, {upper_ci:.4f})")

    # 解释结果
    if len(bootstrap_distribution) > 0:
        if lower_ci < 0 < upper_ci:
            print("结论：中介效应的置信区间包含 0，表明中介效应在统计上不显著。")
        elif not np.isnan(lower_ci):
            print("结论：中介效应的置信区间不包含 0，表明中介效应在统计上显著。")
    else:
        print("结论：无法判断显著性，因为未能计算出有效的 Bootstrap 统计量。")

    print("\n分析完成。")
