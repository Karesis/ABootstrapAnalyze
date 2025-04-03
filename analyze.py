# 导入需要的库
import numpy as np
import pandas as pd
# 导入 statsmodels 用于执行回归分析来估计路径 a 和 b
# 如果没有安装，需要先安装: pip install statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm # <--- 添加这一行

# --- Bootstrap 核心函数 (与之前相同) ---

def bootstrap_statistic(data, statistic_func, n_bootstrap_samples=1000):
  """
  执行 Bootstrap 方法来估计某个统计量的分布。
  (函数内容与之前版本相同，注释省略以保持简洁，请参考上一个回复)
  """
  n_size = data.shape[0] if isinstance(data, (pd.DataFrame, pd.Series)) else len(data)
  bootstrap_stats = []
  print(f"正在生成 {n_bootstrap_samples} 个自助样本并计算中介效应...")
  for i in range(n_bootstrap_samples):
    indices = np.random.choice(np.arange(n_size), size=n_size, replace=True)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        bootstrap_sample = data.iloc[indices]
    else:
        np_data = np.array(data)
        bootstrap_sample = np_data[indices]
    try:
      stat = statistic_func(bootstrap_sample)
      # 检查 stat 是否为 None 或 NaN，如果是则跳过
      if stat is not None and not np.isnan(stat):
          bootstrap_stats.append(stat)
      # else:
      #     print(f"注意：在第 {i+1} 个自助样本上计算得到无效效应值，已跳过。")
    except Exception as e:
      # 如果计算中介效应时出错，打印警告并跳过该样本
      # print(f"警告：在第 {i+1} 个自助样本上计算中介效应时出错: {e}")
      pass # 静默处理错误，或者取消注释上面的print来查看

    if (i + 1) % (n_bootstrap_samples // 10) == 0:
        print(f"已完成 {i + 1}/{n_bootstrap_samples}...")

  print("Bootstrap 重采样完成。")
  valid_stats = np.array(bootstrap_stats) # bootstrap_stats 现在只包含有效值
  if len(valid_stats) < n_bootstrap_samples:
      print(f"警告：最终使用了 {len(valid_stats)} 个有效的自助样本结果（总尝试次数：{n_bootstrap_samples}）。")
  return valid_stats


def calculate_confidence_interval(bootstrap_stats, confidence_level=0.95):
  """
  根据自助法得到的统计量分布计算置信区间（使用百分位法）。
  (函数内容与之前版本相同，注释省略，请参考上一个回复)
  """
  if len(bootstrap_stats) == 0:
      print("错误：没有有效的自助统计量可用于计算置信区间。")
      return (np.nan, np.nan)
  alpha = 1 - confidence_level
  lower_percentile = alpha / 2.0 * 100
  upper_percentile = (1 - alpha / 2.0) * 100
  lower_bound = np.percentile(bootstrap_stats, lower_percentile)
  upper_bound = np.percentile(bootstrap_stats, upper_percentile)
  return lower_bound, upper_bound

# --- 如何使用 ---

# 1. ***** 加载你的真实数据 *****
#    你需要将包含 '体育活动', '健康', '客观环境' 这三列的
#    真实数据文件（比如 .csv 或 .xlsx）加载进来。
#    请替换掉下面的示例数据加载代码。
#    确保你的列名与代码中使用的完全一致！
try:
    # 【【【 在这里替换 'your_real_data.csv' 为你朋友的实际文件名 】】】
    all_data = pd.read_excel('data.xlsx')
    # 如果是 Excel 文件, 用: pd.read_excel('your_real_data.xlsx')

    # 检查必需的列是否存在
    required_cols = ['健康', '体育活动', '客观环境']
    if not all(col in all_data.columns for col in required_cols):
        print(f"错误：数据文件中缺少必需的列。需要包含: {required_cols}")
        # 如果列缺失，可以用下面的示例数据代替来测试代码结构
        raise FileNotFoundError # 触发下面的 except 块

except FileNotFoundError:
    print("错误：找不到指定的数据文件，或文件中列不全。将使用随机生成的示例数据进行演示。")
    print("请务必修改代码以加载您的真实数据文件！")
    # 生成示例数据用于演示
    all_data = pd.DataFrame({
        '健康': np.random.randn(100),      # 自变量 X
        '体育活动': np.random.randn(100),  # 中介变量 M
        '客观环境': np.random.randn(100)   # 因变量 Y
    })
    # 模拟一个中介效应路径
    all_data['体育活动'] = 0.4 * all_data['健康'] + np.random.randn(100) * 0.8
    all_data['客观环境'] = 0.1 * all_data['健康'] + 0.5 * all_data['体育活动'] + np.random.randn(100) * 0.7

print("数据加载完成。数据（或示例数据）前5行：")
print(all_data.head())

print("\n--- 数据检查 ---")
required_cols = ['健康', '体育活动', '客观环境']
for col in required_cols:
    if col in all_data.columns:
        unique_count = all_data[col].nunique()
        variance = all_data[col].var()
        print(f"列 '{col}':")
        print(f"  唯一值数量: {unique_count}")
        print(f"  方差: {variance}")
        if unique_count <= 1:
            print(f"  *** 警告: 列 '{col}' 可能是常量或只有一个唯一值，这会导致回归失败! ***")
    else:
        print(f"  列 '{col}' 不在数据中。")
print("--- 数据检查结束 ---\n")

# 2. ***** 定义计算中介效应 (a*b) 的函数 *****
# ***** 新版本：使用 statsmodels.api.OLS *****
def calculate_indirect_effect(data):
    """
    在给定的数据样本上计算简单中介效应 (a*b)，使用 statsmodels.api.OLS。
    模型: 健康 -> 体育活动 -> 客观环境

    参数:
      data: 一个 pandas DataFrame，包含 X, M, Y 变量的列。

    返回:
      计算出的中介效应值 (a*b)，如果模型拟合失败则返回 None。
    """
    try:
        # 定义变量名 (确认与你的 Excel 文件列名完全一致)
        x_col = '健康'
        m_col = '体育活动'
        y_col = '客观环境'

        # 确保列是数值类型，并处理可能的无穷大值
        for col in [x_col, m_col, y_col]:
            data[col] = pd.to_numeric(data[col], errors='coerce') # 转为数值，无法转换的变 NaN
        data = data.replace([np.inf, -np.inf], np.nan) # 将 Inf/-Inf 替换为 NaN

        # --- 模型 1: 估计 X 对 M 的效应 (路径 a) ---
        # 准备数据 (自动处理 NaN)
        data_a = data[[x_col, m_col]].dropna() # 删除包含 NaN 的行
        if data_a.shape[0] < 2: return None # 如果有效数据太少，无法拟合
        X_a = sm.add_constant(data_a[x_col]) # 给自变量 X 添加常数项 (截距)
        y_a = data_a[m_col]                  # 因变量 M

        # 检查 X_a 的方差是否足够 (避免常量 predictor)
        if X_a.shape[1] > 1 and np.linalg.matrix_rank(X_a) < X_a.shape[1]:
             # print(f"警告: 模型a的 X ('{x_col}') 在此样本中可能存在共线性或近似常量。")
             return None # 如果 X 本身或在添加常数后秩亏，则无法可靠拟合

        model_a = sm.OLS(y_a, X_a).fit()
        path_a = model_a.params[x_col] # 获取 X 的系数 a

        # --- 模型 2: 估计 M 对 Y 的效应 (路径 b)，同时控制 X ---
         # 准备数据 (自动处理 NaN)
        data_b = data[[x_col, m_col, y_col]].dropna() # 删除包含 NaN 的行
        if data_b.shape[0] < 3: return None # 如果有效数据太少（至少需要比变量数多1），无法拟合
        X_b = sm.add_constant(data_b[[m_col, x_col]]) # 给自变量 M 和 X 添加常数项
        y_b = data_b[y_col]                           # 因变量 Y

        # 检查 X_b 的方差/秩是否足够
        if X_b.shape[1] > 1 and np.linalg.matrix_rank(X_b) < X_b.shape[1]:
             # print(f"警告: 模型b的预测变量 ('{m_col}', '{x_col}') 在此样本中可能存在共线性。")
             return None # 如果 M, X 和常数项之间存在完全共线性，则无法拟合

        model_b = sm.OLS(y_b, X_b).fit()
        path_b = model_b.params[m_col] # 获取 M 的系数 b (控制X后)

        # 计算并返回中介效应 (a * b)
        indirect_effect = path_a * path_b
        return indirect_effect

    except Exception as e:
        # 捕获其他可能的错误，例如数据全是NaN，或者statsmodels内部错误
        # 【【【 暂时重新注释掉这行，除非再次出错 】】】
        # print(f"计算中介效应时发生错误: {e}")
        return None
# ***** 函数结束 *****

# 将上面定义的函数赋值给 statistic_to_calculate
statistic_to_calculate = calculate_indirect_effect
print(f"将要计算的统计量： 中介效应 (健康 -> 体育活动 -> 客观环境)")


# 3. ***** 设置 Bootstrap 参数 *****
num_bootstrap_samples = 5000  # 中介效应推荐 >= 5000 次
confidence_level = 0.95    # 95% 置信水平

# 4. ***** 执行 Bootstrap 并计算置信区间 *****
bootstrap_distribution = bootstrap_statistic(all_data, statistic_to_calculate, num_bootstrap_samples)
lower_ci, upper_ci = calculate_confidence_interval(bootstrap_distribution, confidence_level)

# 5. ***** 打印结果 *****
print("\n--- Bootstrap 中介效应结果 ---")
# 计算原始样本的中介效应值
original_indirect_effect = calculate_indirect_effect(all_data)
if original_indirect_effect is not None:
    print(f"原始样本计算出的中介效应值 (a*b): {original_indirect_effect:.4f}")
else:
    print("原始样本计算中介效应失败。")

print(f"通过 Bootstrap 得到的有效中介效应值数量: {len(bootstrap_distribution)}")
print(f"{confidence_level*100:.0f}% 置信区间 (百分位法): ({lower_ci:.4f}, {upper_ci:.4f})")

# 检查置信区间是否包含 0
if len(bootstrap_distribution) > 0:
    if lower_ci < 0 < upper_ci:
        print("中介效应的置信区间包含 0，表明中介效应在统计上不显著。")
    elif not np.isnan(lower_ci):
        print("中介效应的置信区间不包含 0，表明中介效应在统计上显著。")
else:
    print("无法判断显著性，因为没有有效的自助样本结果。")