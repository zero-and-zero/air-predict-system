import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Matplotlib 的字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 加载数据
file_path = 'dataset.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"数据文件 {file_path} 不存在，请检查路径")

try:
    data = pd.read_csv(file_path)
    print(f"成功加载数据，共 {len(data)} 条记录")
except Exception as e:
    raise IOError(f"加载数据文件时出错: {str(e)}")

# 检查必需的列
required_columns = {'AQI指数', 'AQI_1天前', 'PM2.5_1天前', 'PM10_1天前',
                    'So2_1天前', 'No2_1天前', 'O3_1天前', 'Co_1天前'}
missing_columns = required_columns - set(data.columns)
if missing_columns:
    raise ValueError(f"数据中缺少必需的列: {', '.join(missing_columns)}")

# 2. 特征选择
features = ['AQI_1天前', 'PM2.5_1天前', 'PM10_1天前', 'So2_1天前', 'No2_1天前', 'O3_1天前', 'Co_1天前']
X = data[features]
y = data['AQI指数']

# 处理缺失值
if X.isnull().any().any() or y.isnull().any():
    print("警告: 数据中存在缺失值，将使用中位数填充")
    X = X.fillna(X.median())
    y = y.fillna(y.median())

# 3. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"训练集大小: {len(X_train)} 条记录")
print(f"测试集大小: {len(X_test)} 条记录")

# 4. 定义模型列表
models = {
    '随机森林': {
        'model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'desc': "基于多棵决策树的集成学习方法"
    },
    '梯度提升': {
        'model': GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1),
        'desc': "逐步构建模型，每个新模型修正前一个模型的误差"
    },
    '线性回归': {
        'model': LinearRegression(),
        'desc': "假设特征与目标之间为线性关系的简单模型"
    },
    '支持向量机': {
        'model': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'desc': "适用于高维空间中非线性问题的算法"
    },
    'K近邻': {
        'model': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),
        'desc': "基于邻近数据点进行预测的方法"
    }
}

# 创建目录保存模型和评估结果
os.makedirs('models', exist_ok=True)
os.makedirs('evaluation', exist_ok=True)

# 5. 训练并评估模型
results = []
eval_dfs = []

for name, config in models.items():
    model = config['model']
    description = config['desc']

    print(f"\n{'=' * 60}")
    print(f"训练 {name} 模型...")
    print(f"描述: {description}")

    try:
        # 训练模型
        model.fit(X_train, y_train)

        # 保存模型
        model_path = os.path.join('models', f'{name}_model.pkl')
        joblib.dump(model, model_path)
        print(f"✅ 模型已保存至 {model_path}")

        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)

        # 在测试集上评估
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        accuracy = (abs(y_pred - y_test) <= 30).mean()

        # 保存评估结果
        eval_df = pd.DataFrame({
            '真实值': y_test,
            '预测值': y_pred,
            '误差': y_pred - y_test
        })
        eval_path = os.path.join('evaluation', f'{name}_predictions.csv')
        eval_df.to_csv(eval_path, index=False)
        print(f"📊 预测结果已保存至 {eval_path}")

        # 收集性能指标
        model_results = {
            '模型': name,
            '描述': description,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            '准确率(±30)': accuracy,
            '交叉验证RMSE均值': cv_rmse.mean(),
            '交叉验证RMSE标准差': cv_rmse.std()
        }
        results.append(model_results)

        # 打印当前模型结果
        print(f"模型性能:")
        print(f"- MAE: {mae:.2f}")
        print(f"- RMSE: {rmse:.2f}")
        print(f"- R²: {r2:.4f}")
        print(f"- 准确率(误差≤30): {accuracy:.2%}")
        print(f"- 交叉验证RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

    except Exception as e:
        print(f"❌ 训练模型 {name} 时出错: {str(e)}")
        results.append({
            '模型': name,
            '描述': description,
            'MAE': None,
            'RMSE': None,
            'R2': None,
            '准确率(±30)': None,
            '交叉验证RMSE均值': None,
            '交叉验证RMSE标准差': None,
            '错误': str(e)
        })

# 6. 输出模型性能比较
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("模型性能比较:")
print(results_df.drop(columns=['描述']).to_string(index=False))

# 保存详细结果
results_path = os.path.join('evaluation', 'model_performance_summary.csv')
results_df.to_csv(results_path, index=False)
print(f"\n📈 详细评估结果已保存至 {results_path}")


# 7. 可视化结果 - 修复错误并优化
# 7.1 性能指标比较
def plot_metrics_comparison():
    """绘制性能指标比较图"""
    plt.figure(figsize=(14, 10))

    # 创建子图
    metrics = ['MAE', 'RMSE', 'R2', '准确率(±30)']
    titles = ['平均绝对误差(MAE)', '均方根误差(RMSE)', '决定系数(R²)', '准确率(误差≤30)']

    for i, metric in enumerate(metrics):
        ax = plt.subplot(2, 2, i + 1)
        # 使用条形图而不是barplot
        x = range(len(results_df))
        values = results_df[metric]

        # 跳过无效值
        valid_mask = values.notna()
        valid_df = results_df[valid_mask]
        valid_values = valid_df[metric]

        if len(valid_values) > 0:
            bars = ax.bar(x, values, color=plt.cm.tab10.colors[:len(values)])

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                if not pd.isna(height):
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.2f}' if metric != 'R2' else f'{height:.4f}',
                            ha='center', va='bottom')

            # 设置x轴标签
            ax.set_xticks(x)
            ax.set_xticklabels(results_df['模型'], rotation=45)
            ax.set_title(titles[i])
            ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'performance_metrics.png'))
    plt.close()
    print("✅ 性能指标比较图已保存")


# 7.2 交叉验证结果比较 - 修复错误
def plot_cv_comparison():
    """绘制交叉验证结果比较图"""
    plt.figure(figsize=(10, 6))

    # 筛选有效数据
    valid_mask = results_df['交叉验证RMSE均值'].notna() & results_df['交叉验证RMSE标准差'].notna()
    valid_df = results_df[valid_mask]

    if len(valid_df) > 0:
        x = range(len(valid_df))
        means = valid_df['交叉验证RMSE均值']
        stds = valid_df['交叉验证RMSE标准差']

        # 使用条形图而不是barplot
        bars = plt.bar(x, means, yerr=stds, capsize=5, color=plt.cm.tab10.colors[:len(means)])

        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width() / 2., mean + std + 0.5,
                     f'{mean:.2f}±{std:.2f}', ha='center', va='bottom')

        plt.xticks(x, valid_df['模型'], rotation=45)
        plt.title('交叉验证RMSE比较（均值和标准差）')
        plt.ylabel('RMSE')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join('evaluation', 'cross_validation.png'))
        plt.close()
        print("✅ 交叉验证结果比较图已保存")
    else:
        print("⚠️ 无有效的交叉验证数据可绘制图表")


# 7.3 误差分布可视化
def plot_error_distributions():
    """绘制误差分布图"""
    plt.figure(figsize=(14, 10))

    for i, name in enumerate(models.keys()):
        ax = plt.subplot(2, 3, i + 1)

        eval_path = os.path.join('evaluation', f'{name}_predictions.csv')
        if os.path.exists(eval_path):
            try:
                eval_df = pd.read_csv(eval_path)
                errors = eval_df['误差'].dropna()

                if len(errors) > 0:
                    # 绘制直方图和KDE曲线
                    sns.histplot(errors, bins=30, kde=True, color=plt.cm.tab10(i))

                    # 添加统计信息
                    mean_err = errors.mean()
                    std_err = errors.std()
                    ax.axvline(mean_err, color='r', linestyle='--')
                    ax.text(0.95, 0.95, f'均值: {mean_err:.2f}\n标准差: {std_err:.2f}',
                            transform=ax.transAxes, ha='right', va='top',
                            bbox=dict(facecolor='white', alpha=0.8))

                    ax.set_title(f'{name}模型预测误差分布')
                    ax.set_xlabel('误差（预测值 - 真实值）')
                    ax.set_ylabel('频率')
                    ax.grid(True, linestyle='--', alpha=0.5)
            except Exception as e:
                print(f"❌ 无法加载 {name} 的预测结果: {str(e)}")

    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'error_distributions.png'))
    plt.close()
    print("✅ 误差分布图已保存")


# 执行可视化函数
plot_metrics_comparison()
plot_cv_comparison()
plot_error_distributions()

print("\n训练和评估过程完成！")