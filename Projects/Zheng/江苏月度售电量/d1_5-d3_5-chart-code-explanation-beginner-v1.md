# D1.5 / D2.5 / D2.6 / calendar_holiday / D3.5 图表与代码讲解 v1

生成日期：2026-05-23  
最后更新：2026-05-24  
角色：预测建模工程师 / 算法实验员  
用途：给用户逐段理解“代码做了什么、图是怎么来的、每张图应该怎么看、当前说明了什么”

> 本文覆盖 2026-05-23 延期补强阶段和 2026-05-24 D2.6 补强生成的主要 matplotlib 图，共 38 张：D1.5 非线性复核 22 张、D2.5 逐因素证据 5 张、calendar_holiday 1 张、D3.5 试点行业因素分析 5 张、D2.6 逐因素非线性预测验证 5 张。  
> 图像引用的是本机绝对路径，相当于本地“图床”。图像和详细结果仍位于 `output/modeling/` 受保护目录，默认不提交、不推送。

## 0. 先建立读图语言

这一批图分成四类。

第一类是 D1.5 的“分箱形状图”。它不是预测回测图，而是在问：某个因素和某个目标之间，是否可能不是简单直线关系。横轴是因素按大小分成的 4 个分位箱，纵轴是目标变量标准化后的均值，也就是 `target z-score mean`。如果线条不是一直上升或一直下降，而是中间高、两头低，或先降后升，就说明线性相关可能低估了这个因素。

第二类是 D2.5 和 D3.5 的“逐因素 WAPE 改善率总览图”。它们是预测验证图。横轴是相对历史基线的 WAPE 改善率，右侧为改善，左侧为变差。每一根横条是一个因素，颜色只是一级分类，不代表以家族为检验单位。

第三类是 calendar_holiday 的“日历因素改善率图”。它也是预测验证图，但日历因素和经济/天气观测不同：春节、节假日、工作日数这些在预测月份前就已经知道，所以可以直接作为预测时点已知特征进入模型。

第四类是 D2.6 的“非线性相对线性增益差值图”。它不是重新问“这个因素有没有用”，而是在问一个更窄的问题：同一个因素已经做过线性逐因素回测后，如果换成低成本非线性模型，预测误差是否进一步改善。横轴是“最佳非线性模型相对线性单因素 Ridge 的 WAPE 改善差值”，右侧代表非线性优于线性，左侧代表非线性不如线性。

几个关键概念：

- `z-score`：把一列数减去平均值再除以标准差。它不保留原始业务数值，只表达“比平均水平高多少个标准差”。
- `lag=0`：同期关系，只能解释“同月一起变化”，不能直接做真实预测。
- `lag=1/2/3/12`：因素滞后 1、2、3 或 12 个月，用于判断预测时点更可能拿得到的信息。
- `WAPE`：加权绝对百分误差，越低越好。
- `WAPE 改善率`：`(历史基线 WAPE - 加入因素后 WAPE) / 历史基线 WAPE`。大于 0 表示比历史基线好，小于 0 表示变差。
- `Ridge`：带 L2 正则的线性回归。D2.5 里曾用 `numpy` 手写闭式解；D2.6 起按“有成熟轮子就用成熟轮子”的原则，改用 `scikit-learn` 的 `Ridge`、`Pipeline`、`PolynomialFeatures`、浅层树和随机森林等成熟组件。

## 1. D1.5 非线性复核

### 1.1 代码在做什么

D1.5 脚本是 [08_run_d1_5_nonlinear_review.py](/Users/neonho/Documents/GitHub/js-power-forecast/scripts/mvp/08_run_d1_5_nonlinear_review.py)。

核心参数：

```python
LAGS = (0, 1, 2, 3, 12)
TRANSFORMS = ("level", "diff12", "month_demeaned")
MIN_OBS = 18
```

意思是：同一个目标和同一个因素，要分别尝试同期、1 月滞后、2 月滞后、3 月滞后、12 月滞后；也要分别看原始水平、同比差分、去月份均值后的残差。这样做是为了避免只看一条简单相关线。

构造目标和因素配对的代码切片：

```python
def make_pair(group, feature, *, lag, transform):
    y = pd.to_numeric(group[TARGET_COL], errors="coerce")
    x = pd.to_numeric(group[feature], errors="coerce")
    if transform == "level":
        y_t = y
        x_t = x
    elif transform == "diff12":
        y_t = y.diff(12)
        x_t = x.diff(12)
    elif transform == "month_demeaned":
        month = group["_month_num"]
        y_t = y - y.groupby(month).transform("mean")
        x_t = x - x.groupby(month).transform("mean")
    return SeriesPair(target=y_t, feature=x_t.shift(lag), period=group["_timestamp"])
```

你可以这样理解：

- `level`：看原始月度水平之间的关系。
- `diff12`：看“相比 12 个月前变化了多少”，更接近同比变化。
- `month_demeaned`：先去掉每个月自己的平均季节性，再看剩余波动。
- `x_t.shift(lag)`：把因素往后挪，比如 `lag=2` 就是用两个月前的因素解释当前目标。

非线性证据不是只靠二次曲线。脚本同时看：

```python
quadratic_r2_gain(...)
discretized_mutual_information(...)
binned_target_z_range(...)
segmented_slope_shift(...)
bin_shape(...)
```

对应含义：

- `quadratic_r2_gain`：加一个平方项后解释力是否明显提升，只作辅助。
- `discretized_mutual_information`：把目标和因素都分箱，看它们是否有信息关联。
- `binned_target_z_range`：不同因素分箱下，目标均值差得大不大。
- `segmented_slope_shift`：低因素区间和高因素区间的斜率是否明显不同。
- `bin_shape`：分箱均值是否非单调。

D1.5 图就是在 `plot_binned_shape` 里画出来的：

```python
data["bin"] = pd.qcut(data["feature"], q=4, labels=False, duplicates="drop")
grouped = data.assign(target_z=zscore(data["target"])).groupby("bin")["target_z"].mean()
ax.plot(grouped["bin"].astype(int) + 1, grouped["target_z"], marker="o")
```

横轴不是月份，也不是原始因素值，而是因素从小到大分成的 4 桶。纵轴是每桶里目标的 z-score 均值。

### 1.2 D1.5 图怎么看

看这类图时，先问四个问题：

1. 线是不是基本单调？如果不是，线性回归可能会低估关系。
2. 四个分箱之间的高低差是否明显？差越大，因素分组和目标水平越可能有关。
3. `lag` 是多少？`lag=0` 只能解释，不能直接预测。
4. `transform` 是什么？如果是 `diff12` 或 `month_demeaned`，说明关系主要体现在同比变化或去季节残差上。

### 图 1：城乡居民生活用电合计 vs 平均最高气温

![图1](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/001_binned_shape_B_城乡居民生活用电合计_weather_avg_平均最高气温.png)

- 代码来源：`make_pair(..., transform="level", lag=0)` 后进入 `plot_binned_shape`。
- 读法：横轴是平均最高气温的 4 个分位箱，纵轴是居民生活用电目标的标准化均值。
- 当前说明：脚本将它标为 `linearly_weak_but_nonlinear_strong`，且是 `non_monotone`。这很符合居民用电的直觉：气温和居民用电常常不是一条直线，低温取暖和高温制冷都可能推高用电。
- 边界：`lag=0`，只能作同期解释，不能直接当预测特征；若要预测，需要天气预报或滞后天气方案。

### 图 2：第三产业 vs 进出口总额同比

![图2](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/002_binned_shape_第三产业_econ_imports_exports_total_进出口总额_同比_百分比.png)

- 代码来源：`transform="level"`，`lag=2`。
- 读法：横轴是两个月前进出口同比的分箱，纵轴是第三产业目标 z-score 均值。
- 当前说明：被标为线性弱但非线性较强，提示外贸景气和第三产业用电之间可能不是简单同比越高用电越高。
- 边界：这是候选线索，需要 D2.5 的逐因素回测判断是否真正改善预测。

### 图 3：城乡居民生活用电合计 vs 月份编号

![图3](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/003_binned_shape_B_城乡居民生活用电合计_month_num.png)

- 代码来源：`month_num`，`transform="level"`，`lag=2`。
- 读法：月份编号被分成 4 桶后，看居民用电目标均值。
- 当前说明：月份本身呈非线性季节结构，居民用电通常有夏季、冬季双峰，线性月份编号当然表达不好。
- 边界：这张图说明“月份季节性不是线性的”，不是说 `month_num` 本身就是好预测特征。实际预测中更常用 `month_sin/month_cos`、节假日和春节错位字段。

### 图 4：第一产业 vs 进出口总额同比

![图4](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/004_binned_shape_第一产业_econ_imports_exports_total_进出口总额_同比_百分比.png)

- 代码来源：`transform="month_demeaned"`，`lag=2`。
- 读法：先扣掉月份平均季节性，再看两个月前进出口同比分箱和第一产业残差之间的关系。
- 当前说明：非线性证据较强，说明这个候选关系不只是季节性带来的。
- 边界：第一产业样本波动可能较大，后续不能只凭这张图包装成经济因素主线。

### 图 5：第三产业 vs 月份编号

![图5](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/005_binned_shape_第三产业_month_num.png)

- 代码来源：`month_num`，`transform="level"`，`lag=12`。
- 读法：这里的 `lag=12` 本质上仍在看年度季节结构。
- 当前说明：第三产业也存在非线性月份季节性，不能用简单线性月份编号概括。
- 边界：季节性最好和日历因素拆开解释，尤其春节错位不能只靠月份编号。

### 图 6：第一产业 vs 10 米主导风向均值

![图6](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/006_binned_shape_第一产业_weather_avg_10米主导风向均值.png)

- 代码来源：`transform="level"`，`lag=2`。
- 读法：横轴是两个月前风向指标分箱。
- 当前说明：脚本识别到非单调分箱形状。它更像天气综合条件的代理线索，而不是风向本身有直接经济含义。
- 边界：风向变量业务解释较弱，建议只作为天气候选池的一部分，不宜单独对外强调。

### 图 7：第三产业 vs 季度

![图7](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/007_binned_shape_第三产业_quarter.png)

- 代码来源：`quarter`，`transform="level"`，`lag=3`。
- 读法：季度分组后看第三产业目标均值。
- 当前说明：季度是粗季节特征，非线性形状说明季度之间用电水平不同。
- 边界：季度太粗，不能替代月份、节假日、春节错位等更细日历字段。

### 图 8：全社会用电总计 vs 进出口总额同比

![图8](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/008_binned_shape_全社会用电总计_econ_imports_exports_total_进出口总额_同比_百分比.png)

- 代码来源：`transform="month_demeaned"`，`lag=12`。
- 读法：看去季节残差层面，去年同期附近的进出口同比分箱是否对应不同总量用电水平。
- 当前说明：外贸因素对总量可能有非线性或分段线索。
- 边界：`lag=12` 的解释容易混入年度周期，需要和历史目标滞后基线比较后再判断增益。

### 图 9：全社会用电总计 vs 10 米风速均值

![图9](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/009_binned_shape_全社会用电总计_weather_avg_10米风速均值.png)

- 代码来源：`transform="level"`，`lag=0`。
- 当前说明：同期风速和总量用电存在非线性形状线索，但风速自身不一定有直接稳定机制。
- 边界：`lag=0` 只解释，不预测。若要预测，应换成天气预报或滞后天气。

### 图 10：第三产业 vs 日照时长合计

![图10](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/010_binned_shape_第三产业_weather_avg_日照时长合计_小时.png)

- 代码来源：`transform="diff12"`，`lag=1`。
- 当前说明：同比变化层面，日照时长与第三产业用电变化存在非线性线索。
- 边界：这是滞后天气候选，可以进入 D2.5 或行业验证，但仍要看预测误差是否改善。

### 图 11：第一产业 vs CPI 上月等于 100

![图11](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/011_binned_shape_第一产业_econ_cpi_CPI_上月等于100.png)

- 代码来源：`transform="diff12"`，`lag=3`。
- 当前说明：CPI 环比指标和第一产业用电变化可能有非线性关联。
- 边界：CPI 对第一产业的机制不如天气直接，需谨慎解释为宏观环境线索，而不是单一驱动因素。

### 图 12：全社会用电总计 vs 10 米最大风速均值

![图12](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/012_binned_shape_全社会用电总计_weather_avg_10米最大风速均值.png)

- 代码来源：`transform="diff12"`，`lag=2`。
- 当前说明：同比变化层面出现非单调分箱线索。
- 边界：风速类指标解释性较弱，建议和其他天气字段一起作为候选，不单独包装。

### 图 13：全社会用电总计 vs 平均云量

![图13](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/013_binned_shape_全社会用电总计_weather_avg_平均云量.png)

- 代码来源：`transform="level"`，`lag=12`。
- 当前说明：平均云量在总量层级有分箱差异，可能体现季节、光照和天气状态的综合代理。
- 边界：`lag=12` 可能和季节重复，预测价值需要看 D2.5。

### 图 14：第三产业 vs 平均云量

![图14](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/014_binned_shape_第三产业_weather_avg_平均云量.png)

- 代码来源：`transform="level"`，`lag=1`。
- 当前说明：第三产业对天气状态可能存在滞后一月的非线性线索。
- 边界：仍是候选关系，不说明因果。

### 图 15：第二产业 vs 平均云量

![图15](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/015_binned_shape_第二产业_weather_avg_平均云量.png)

- 代码来源：`transform="level"`，`lag=1`。
- 当前说明：第二产业也出现平均云量分箱差异，但工业用电的机制可能更受生产安排、经济景气和行业结构影响。
- 边界：不要把云量解释成第二产业直接驱动因素，它更可能是天气综合代理。

### 图 16：第二产业 vs PPI 购进上月等于 100

![图16](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/016_binned_shape_第二产业_econ_ppi_PPI购进_上月等于100.png)

- 代码来源：`transform="diff12"`，`lag=0`。
- 当前说明：生产资料价格变化和第二产业用电变化有同期非线性线索。
- 边界：`lag=0` 只作解释；同月 PPI 在预测时点通常不可用，不能直接进入预测验证。

### 图 17：第一产业 vs PPI 购进上月等于 100

![图17](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/017_binned_shape_第一产业_econ_ppi_PPI购进_上月等于100.png)

- 代码来源：`transform="diff12"`，`lag=2`。
- 当前说明：PPI 购进指标对第一产业也出现滞后非线性线索，但机制需要业务复核。
- 边界：这类宏观价格指标不能混成“经济家族整体有效”，必须逐字段看。

### 图 18：城乡居民生活用电合计 vs 运行容量

![图18](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/018_binned_shape_B_城乡居民生活用电合计_运行容量.png)

- 代码来源：`transform="month_demeaned"`，`lag=0`。
- 当前说明：去月份均值后，居民用电和运行容量存在分箱差异。
- 边界：内部业务同月值不直接用于预测；还要警惕容量和售电量之间的口径、业务流程和同步统计关系。

### 图 19：城乡居民生活用电合计 vs 平均云量

![图19](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/019_binned_shape_B_城乡居民生活用电合计_weather_avg_平均云量.png)

- 代码来源：`transform="diff12"`，`lag=3`。
- 当前说明：居民用电同比变化和三个月前云量存在非线性线索。
- 边界：天气滞后是否有稳定预测价值，要看 D2.5；本图只说明形状值得复核。

### 图 20：城乡居民生活用电合计 vs 10 米阵风最大值均值

![图20](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/020_binned_shape_B_城乡居民生活用电合计_weather_avg_10米阵风最大值均值.png)

- 代码来源：`transform="level"`，`lag=0`。
- 当前说明：同期阵风指标和居民用电存在分箱差异。
- 边界：同期天气观测不能直接预测；且阵风机制不如温度直观。

### 图 21：全社会用电总计 vs 10 米阵风最大值均值

![图21](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/021_binned_shape_全社会用电总计_weather_avg_10米阵风最大值均值.png)

- 代码来源：`transform="level"`，`lag=2`。
- 当前说明：两个月前阵风指标和总量用电存在非线性分箱线索。
- 边界：天气风类指标应作为候选而非主解释变量。

### 图 22：城乡居民生活用电合计 vs 有效天数

![图22](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d1_5_nonlinear_review/matplotlib_charts/022_binned_shape_B_城乡居民生活用电合计_weather_avg_有效天数.png)

- 代码来源：`transform="level"`，`lag=3`。
- 当前说明：脚本保留它是因为二次项辅助增益较明显，但分箱形状标记为 `insufficient`，说明图形证据不如前面那些稳定。
- 边界：这张图更像“需要人工复核”的候选，不应直接进入主结论。

## 2. D2.5 逐因素证据复核

### 2.1 代码在做什么

D2.5 脚本是 [09_run_d2_5_single_factor_evidence.py](/Users/neonho/Documents/GitHub/js-power-forecast/scripts/mvp/09_run_d2_5_single_factor_evidence.py)。

它先给目标变量造历史特征：

```python
for lag in TARGET_LAGS:
    frame[f"target_lag_{lag}"] = y.shift(lag)
for window in (3, 6, 12):
    frame[f"target_ma_{window}"] = y.shift(1).rolling(window=window).mean()
frame["month_sin"] = np.sin(month_angle)
frame["month_cos"] = np.cos(month_angle)
```

这里的历史基线包括：目标滞后 `1/2/3/12` 月、目标移动平均 `3/6/12` 月、月份周期项。

然后给每个因素造滞后特征：

```python
for feature in feature_cols:
    numeric = pd.to_numeric(frame[feature], errors="coerce")
    for lag in FACTOR_LAGS:
        lagged_data[f"{feature}__lag_{lag}"] = numeric.shift(lag)
```

这就是你前面问的“滞后有没有试”：这里每个因素都尝试了 `t-1/t-2/t-3/t-12`。同月因素没有进入预测验证。

逐因素实验的设计是：

```python
feature_columns = tuple([*history_cols, *lagged_cols])
run_ridge_backtest(..., "single_factor_lagged", feature, feature_columns)
```

也就是说，每次只加一个因素的滞后项，再和纯历史基线比较。不是把所有因素一起塞进去，也不是按家族挑少数字段。

Ridge 的核心预测代码：

```python
coef = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_centered)
return float((y_mean + x_test @ coef)[0])
```

这是 `numpy` 实现的 Ridge 闭式解，不是 `sklearn`。

图表生成逻辑：

```python
data["plot_value"] = data["wape_improvement_vs_history"] * 100
ax.barh(labels, data["plot_value"], color=colors)
ax.axvline(0, color="#444444")
```

所以读图时只要看 0 线：右侧为比历史基线更好，左侧为更差。

### 图 23：城乡居民生活用电合计逐因素回测总览

![图23](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_5_single_factor_evidence/matplotlib_charts/001_single_factor_overview_B_城乡居民生活用电合计.png)

- 横轴：相对历史基线 WAPE 改善率，右侧为改善，左侧为变差。
- 纵轴：每个单独因素。
- 颜色：一级分类，只是展示标签。
- 当前结果：31 个因素未改善，8 个改善，7 个轻微改善或持平。
- 目前最靠右的线索包括：分布式光伏自发自用电量、分布式光伏发电量、PPI 出厂累计涨幅。
- 怎么看：这说明居民侧不是所有理论因素都有预测增益。新型负荷和部分价格指标出现线索，但不能直接写成稳定主线。

### 图 24：全社会用电总计逐因素回测总览

![图24](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_5_single_factor_evidence/matplotlib_charts/002_single_factor_overview_全社会用电总计.png)

- 当前结果：32 个因素未改善，5 个改善，9 个轻微改善或持平。
- 目前最靠右的线索包括：用户数、进出口总额绝对量、天气有效天数。
- 怎么看：总量层级的历史基线已经很强，所以很多因素加进去不一定改善。能改善的因素更值得进入下一轮候选池。
- 边界：用户数属于内部业务字段，需要继续确认预测时点可得性和统计口径。

### 图 25：第一产业逐因素回测总览

![图25](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_5_single_factor_evidence/matplotlib_charts/003_single_factor_overview_第一产业.png)

- 当前结果：22 个因素未改善，10 个改善，14 个轻微改善或持平。
- 目前最靠右的线索包括：降雨量、降水量、社会消费品零售总额当月同比。
- 怎么看：第一产业对天气更敏感，这与业务直觉一致；天气滞后因素在这里更值得保留。
- 边界：社会消费品零售总额对第一产业的机制不直接，应该作为统计线索，不宜强解释。

### 图 26：第三产业逐因素回测总览

![图26](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_5_single_factor_evidence/matplotlib_charts/004_single_factor_overview_第三产业.png)

- 当前结果：32 个因素未改善，7 个改善，7 个轻微改善或持平。
- 目前最靠右的线索包括：社会消费品零售总额当月同比、工业增加值当月同比、PPI 出厂上月等于 100。
- 怎么看：第三产业里经济景气字段出现一些改善线索，但字段差异很大，不能写成“经济因素整体有效”。
- 边界：这些公开经济指标的发布时间需要继续结构化，预测时只能用滞后口径。

### 图 27：第二产业逐因素回测总览

![图27](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_5_single_factor_evidence/matplotlib_charts/005_single_factor_overview_第二产业.png)

- 当前结果：36 个因素未改善，5 个改善，5 个轻微改善或持平。
- 目前最靠右的线索包括：进出口总额绝对量、天气有效天数、工业增加值当月同比。
- 怎么看：第二产业确实出现了外贸、工业景气、天气等候选线索，但整体改善数量少，说明历史基线和产业自身惯性仍然很重要。
- 边界：不能因为少数字段有效就把整类经济因素都打包成有效。

## 3. calendar_holiday 轻量验证

### 3.1 代码在做什么

calendar 脚本是 [10_run_calendar_holiday_ablation.py](/Users/neonho/Documents/GitHub/js-power-forecast/scripts/mvp/10_run_calendar_holiday_ablation.py)。

它优先读取数据工程师交付的官方调休口径表：

```python
CALENDAR_INPUT = Path("data/model-ready/calendar/mvp/calendar_holiday_monthly.csv")
if CALENDAR_INPUT.exists():
    frame = pd.read_csv(CALENDAR_INPUT)
    frame["_period"] = pd.PeriodIndex(frame["month"].astype(str), freq="M")
```

使用的日历字段包括：

```python
CALENDAR_FEATURES = (
    "days_in_month",
    "workday_count",
    "weekend_count",
    "public_holiday_count",
    "spring_festival_days_in_month",
    "spring_festival_month_flag",
    "pre_spring_festival_flag",
    "post_spring_festival_flag",
    "workday_ratio",
)
```

关键区别：这些字段在预测月份前可以知道，所以它们不像同月经济数据或同月天气观测那样必须滞后。

### 图 28：calendar_holiday 相对历史基线改善率

![图28](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_5_calendar_holiday_ablation/matplotlib_charts/001_calendar_holiday_wape_improvement.png)

- 横轴：加入 `calendar_holiday` 后相对历史基线的 WAPE 改善率。
- 当前结果：全社会用电总计、第一产业、第二产业、第三产业改善；城乡居民生活用电合计未改善。
- 怎么看：日历因素不是简单月份周期项，它补充了工作日数、春节分布、节前节后窗口等结构。
- 当前说明：`calendar_holiday` 是当前最值得保留的预测时点已知因素之一。
- 边界：居民侧未改善，说明居民用电可能更多受到温度、季节双峰和居家行为影响，不能假设所有层级都同样受日历因素改善。

## 4. D3.5 试点行业因素分析

### 4.1 代码在做什么

D3.5 脚本是 [11_run_d3_5_industry_factor_analysis.py](/Users/neonho/Documents/GitHub/js-power-forecast/scripts/mvp/11_run_d3_5_industry_factor_analysis.py)。

它读取 5 个试点行业宽表：

```python
INPUT_PANEL = Path("data/model-ready/experiments/mvp/d3_5_trial_industry_panel.csv")
```

行业层级的特征构造和 D2.5 类似：

```python
for lag in TARGET_LAGS:
    frame[f"target_lag_{lag}"] = y.shift(lag)
for feature in feature_cols:
    for lag in FACTOR_LAGS:
        lagged_data[f"{feature}__lag_{lag}"] = numeric.shift(lag)
```

也就是说，它同样使用目标历史滞后和因素滞后。行业分析还有一个额外的滞后相关筛查：

```python
for lag in (0, 1, 2, 3, 12):
    data = pd.DataFrame({"target": y, "feature": x0.shift(lag)}).dropna()
    corr = data["target"].rank().corr(data["feature"].rank())
```

这张筛查表不是最终预测结论，而是帮助我们看行业层级是否存在更明显的领先/滞后关系。

### 图 29：化学原料和化学制品制造业逐因素回测总览

![图29](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d3_5_industry_factor_analysis/matplotlib_charts/001_industry_factor_overview_化学原料和化学制品制造业.png)

- 当前结果：39 个因素未改善，12 个改善，5 个轻微改善或持平。
- 目前最靠右的线索包括：居民充电设施电量、平均体感温度、PPI 购进累计涨幅。
- 怎么看：这个行业出现了天气、价格和新型负荷线索，但新型负荷是否有直接机制需要业务复核。
- 边界：行业名称来自当前内部口径，正式入稿前仍需甲方行业映射表确认。

### 图 30：汽车制造业逐因素回测总览

![图30](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d3_5_industry_factor_analysis/matplotlib_charts/002_industry_factor_overview_汽车制造业.png)

- 当前结果：35 个因素未改善，9 个改善，12 个轻微改善或持平。
- 目前最靠右的线索包括：工作日比例、公共节假日天数、春节前标记。
- 怎么看：汽车制造业试点里，日历和工作日结构比较突出，说明生产节奏可能受节假日安排影响。
- 边界：这仍是试点行业观察，不是全制造业结论。

### 图 31：纺织业逐因素回测总览

![图31](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d3_5_industry_factor_analysis/matplotlib_charts/003_industry_factor_overview_纺织业.png)

- 当前结果：40 个因素未改善，11 个改善，5 个轻微改善或持平。
- 目前最靠右的线索包括：春节前标记、社会消费品零售总额绝对量、进出口总额绝对量。
- 怎么看：纺织业可能同时受节前生产/订单安排和外需、消费景气影响。
- 边界：由于样本短，不能据此写成稳定行业规律。

### 图 32：计算机、通信和其他电子设备制造业逐因素回测总览

![图32](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d3_5_industry_factor_analysis/matplotlib_charts/004_industry_factor_overview_计算机_通信和其他电子设备制造业.png)

- 当前结果：41 个因素未改善，9 个改善，6 个轻微改善或持平。
- 目前最靠右的线索包括：春节前标记、工作日比例、公共节假日天数。
- 怎么看：这个行业试点里，日历因素明显靠前，说明生产和交付节奏可能对工作日结构较敏感。
- 边界：内部业务字段如户数、运行容量在本轮没有表现出改善，不能强行解释。

### 图 33：黑色金属冶炼和压延加工业逐因素回测总览

![图33](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d3_5_industry_factor_analysis/matplotlib_charts/005_industry_factor_overview_黑色金属冶炼和压延加工业.png)

- 当前结果：36 个因素未改善，11 个改善，9 个轻微改善或持平。
- 目前最靠右的线索包括：日照时长、春节前标记、公共节假日天数。
- 怎么看：这个行业出现天气和日历线索，但同时大量因素仍不改善，说明行业用电有强自身惯性或受未接入事件影响。
- 边界：固定资产投资等部分经济字段未改善，不代表所有经济指标无效，只代表本轮字段、滞后和样本下不支持。

## 5. D2.6 逐因素非线性预测验证

### 5.1 为什么要补 D2.6

你前面提出的质疑非常关键：D1.5 虽然做了非线性形态摸底，但它还不是预测回测；D2.5 虽然做了逐因素预测回测，但它用的是线性 Ridge。两者中间缺了一个问题：

> 如果同一个因素在线性模型里没表现好，会不会只是因为线性模型太简单？  
> 如果换成低成本非线性模型，它是否能真正改善预测误差？

D2.6 就是为了回答这个问题。

可以把 D1.5、D2.5、D2.6 的关系理解成三层筛子：

| 阶段 | 回答的问题 | 是否预测回测 | 是否逐因素 | 是否非线性 |
| --- | --- | --- | --- | --- |
| D1.5 | 因素和目标之间有没有非线性形态线索 | 否 | 是 | 是，形态摸底 |
| D2.5 | 加入单个因素滞后项后，线性 Ridge 是否改善历史基线 | 是 | 是 | 否，线性验证 |
| D2.6 | 同一个因素换成非线性项或非线性模型后，是否比线性单因素更好 | 是 | 是 | 是，预测验证 |

所以 D2.6 不是推翻 D1.5 或 D2.5，而是把两者接起来。它最重要的价值是区分两类情况：

1. D1.5 看起来有非线性形态，但 D2.6 严格预测回测后没有改善。这说明“有形状”不等于“能预测”。
2. D2.5 线性单因素弱，但 D2.6 非线性模型能改善。这说明线性模型可能低估了这个因素。

### 5.2 D2.6 用了哪些成熟工具

D2.6 脚本是 [13_run_d2_6_single_factor_nonlinear_evidence.py](/Users/neonho/Documents/GitHub/js-power-forecast/scripts/mvp/13_run_d2_6_single_factor_nonlinear_evidence.py)。

项目经理原建议脚本编号是 `12`，但项目里已经有 [12_export_explanation_markdown_pdf.py](/Users/neonho/Documents/GitHub/js-power-forecast/scripts/mvp/12_export_explanation_markdown_pdf.py)，所以这次用了 `13`，避免覆盖已有脚本。

这次也按你的原则做了修正：不再手写 sklearn 已经成熟覆盖的东西。使用的依赖和组件是：

```python
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
```

这些组件分别负责：

- `Pipeline`：把“缺失值填补、标准化、特征变换、模型训练”串成一个整体。好处是每个回测窗口都重新 `fit`，不容易把测试月信息泄漏进训练过程。
- `SimpleImputer`：只用训练窗口估计中位数，再填补训练集和测试月中的缺失特征。
- `StandardScaler`：只用训练窗口估计均值和标准差，再标准化测试月。
- `PolynomialFeatures`：给单个因素的滞后项构造二次项和交互项，用来捕捉平滑非线性。
- `Ridge`：线性回归加 L2 正则，作为历史基线、线性单因素和二次项模型的回归器。
- `DecisionTreeRegressor(max_depth=2)`：浅层树模型，用少量阈值切分捕捉“超过某个水平后关系改变”的模式。
- `RandomForestRegressor(max_depth=3)`：低复杂度随机森林，作为浅层树的敏感性对照。

安装和验证记录在工作台摘要里：

```bash
uv add scikit-learn
uv run python -c "import sklearn; print(sklearn.__version__)"
```

当前版本是 `1.8.0`。这不是异常申请，而是正常成熟工具选型。

### 5.3 D2.6 的实验对象是什么

核心常量如下：

```python
TEST_START = pd.Period("2024-07", freq="M")
MIN_TRAIN_OBS = 24
MIN_FACTOR_OBS = 18
TARGET_LAGS = (1, 2, 3, 12)
FACTOR_LAGS = (1, 2, 3, 12)
```

这几行定义了实验边界：

- `TEST_START = 2024-07`：从 2024 年 7 月开始做回测。
- `MIN_TRAIN_OBS = 24`：训练窗口至少要有 24 个有效月份。
- `MIN_FACTOR_OBS = 18`：因素字段至少要有 18 个非空样本才进入候选。
- `TARGET_LAGS = (1, 2, 3, 12)`：历史目标使用上 1、2、3、12 个月。
- `FACTOR_LAGS = (1, 2, 3, 12)`：因素也只使用上 1、2、3、12 个月。

这意味着 D2.6 仍然遵守 D2.5 的预测边界：公开经济、天气观测、内部业务、新型负荷的同月值不进入预测验证。它只看滞后值。

这次覆盖：

- 5 个首轮目标层级。
- 46 个因素字段。
- 230 个 `target_name + feature_name` 字段级组合。

这里的 230 不是 5 × 46 后再乘以模型数量。230 指的是“目标和因素的组合数”。每个组合内部又会比较多个模型。

### 5.4 每个因素比较了哪些模型

脚本用 `ModelSpec` 把每种模型登记清楚：

```python
MODEL_SPECS = (
    ModelSpec("history_baseline", "sklearn_ridge_history", "ridge", False),
    ModelSpec("linear_single_factor_lagged", "sklearn_ridge_linear_lags", "ridge", True),
    ModelSpec("nonlinear_single_factor_poly2_ridge", "sklearn_poly2_ridge_factor_lags", "poly2_ridge", True),
    ModelSpec("nonlinear_single_factor_tree_depth2", "sklearn_decision_tree_depth2", "tree_depth2", True),
    ModelSpec("nonlinear_single_factor_rf_depth3", "sklearn_random_forest_depth3", "rf_depth3", True),
)
```

逐个解释：

1. `history_baseline`：只用历史目标、滚动均值和月份周期。它回答“只看自己过去，能预测到什么程度”。
2. `linear_single_factor_lagged`：历史基线 + 某一个因素的滞后项，用 Ridge 线性回归。它对应 D2.5 的主思想。
3. `nonlinear_single_factor_poly2_ridge`：历史基线 + 某一个因素的滞后项 + 这个因素滞后项的二次多项式项，再用 Ridge。它是平滑非线性。
4. `nonlinear_single_factor_tree_depth2`：历史基线 + 某一个因素滞后项，用最大深度为 2 的浅层树。它是阈值型非线性。
5. `nonlinear_single_factor_rf_depth3`：历史基线 + 某一个因素滞后项，用最大深度为 3 的低复杂度随机森林。它是更柔一点的树模型对照。

要注意：这不是把所有因素一起塞进去。每次仍然只考察一个因素，只是用几种模型形态去考察同一个因素。

### 5.5 D2.6 如何构造训练集和测试集

回测的核心函数是 `run_sklearn_backtest`。关键片段如下：

```python
for period in test_periods(frame):
    train = frame[frame["_period"] < period].copy()
    test = frame[frame["_period"] == period].copy()
```

这两行非常重要：

- 对每一个测试月，只拿它之前的月份当训练集。
- 当前测试月之后的数据完全不参与训练。
- 这就是 expanding-window，也叫扩展窗口回测。

之后脚本会筛掉目标为空的训练行：

```python
train_y = train[TARGET_COL].map(to_float)
train_mask = train_y.notna()
train = train.loc[train_mask].copy()
train_y = train_y.loc[train_mask]
```

再检查训练样本够不够：

```python
if len(train) < MIN_TRAIN_OBS or test.empty:
    continue
```

所以 D2.6 不会在样本太短时硬做判断。

### 5.6 D2.6 如何选择特征列

对于每个因素，脚本先构造历史目标特征：

```python
history_cols = history_feature_columns(prepared)
```

历史列包括：

- `target_lag_1`
- `target_lag_2`
- `target_lag_3`
- `target_lag_12`
- `target_ma_3`
- `target_ma_6`
- `target_ma_12`
- `month_sin`
- `month_cos`

然后对当前因素构造滞后列：

```python
lagged_cols = lagged_columns_for(prepared, feature)
```

因素滞后列类似：

```text
某因素__lag_1
某因素__lag_2
某因素__lag_3
某因素__lag_12
```

如果某个滞后列在训练窗口里有效值太少，脚本会丢掉它：

```python
if train_col.notna().sum() < max(8, MIN_TRAIN_OBS // 3):
    continue
```

如果某个特征在训练窗口里没有变化，标准差为 0，也会丢掉：

```python
std = float(np.std(filled.to_numpy(dtype=float)))
if std == 0 or math.isnan(std):
    continue
```

这一步的目的不是挑“表现好”的因素，而是防止无效列进入模型。它只看训练窗口内的可用性和方差，不看测试月结果。

### 5.7 为什么二次项 Ridge 不是“把所有东西都平方”

如果把历史目标、月份周期和因素滞后项全部放进 `PolynomialFeatures`，就会出现一个问题：非线性提升可能来自历史目标的平方项，而不是当前因素的非线性。

D2.6 脚本为了尽量保持“单因素非线性”的解释边界，用了 `ColumnTransformer`：

```python
transformers.append(
    (
        "history",
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        ),
        list(history_cols),
    )
)
```

历史目标这部分只做填补和标准化，不做二次项。

因素滞后项这部分才做二次多项式：

```python
transformers.append(
    (
        "factor_poly2",
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("poly2", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
            ]
        ),
        list(factor_lag_cols),
    )
)
```

最后再接 Ridge：

```python
return Pipeline(
    [
        ("preprocess", ColumnTransformer(transformers=transformers, remainder="drop")),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ]
)
```

这就保证：D2.6 的“二次项非线性”主要是在考察这个单个因素的滞后项，而不是把整个历史基线也变成复杂非线性。

### 5.8 浅层树和随机森林为什么要控制复杂度

树模型可以表达阈值关系，比如：

- 降水量很低和很高时，用电反应不同。
- 气温超过某个区间后，用电增长明显。
- 某个经济指标在低景气和高景气时，对用电的关系不同。

但月度样本很短，如果树太深，很容易记住噪声。因此 D2.6 使用浅层树：

```python
DecisionTreeRegressor(
    max_depth=2,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
```

这表示：

- 最多切两层，结构非常浅。
- 每个叶子至少 5 个样本，避免一个叶子只记住一两个月。
- 固定随机种子，保证可复现。

随机森林也限制复杂度：

```python
RandomForestRegressor(
    n_estimators=60,
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
```

这里没有追求最高精度，而是在做低成本敏感性验证：如果浅层非线性模型都能稳定改善，才说明后续值得继续复核。

### 5.9 D2.6 怎么判断“非线性有没有价值”

脚本先计算每种模型相对历史基线的 WAPE 改善率：

```python
frame.loc[idx, "wape_improvement_vs_history"] = (
    history_wape - frame.loc[idx, "wape"]
) / history_wape
```

这个数的含义是：

- 大于 0：比历史基线好。
- 小于 0：比历史基线差。
- 越大越好。

然后，对每个 `target_name + feature_name`，脚本取：

- 线性模型的改善率：`best_linear_wape_improvement`
- 最好的非线性模型改善率：`best_nonlinear_wape_improvement`
- 两者差值：`nonlinear_minus_linear`

差值的含义是：

```text
nonlinear_minus_linear
= 最佳非线性模型相对历史基线的改善率
- 线性单因素模型相对历史基线的改善率
```

如果它大于 0，说明非线性比线性好。  
如果它小于 0，说明非线性比线性差。

但这里有一个很容易误读的点：

> 非线性比线性好，不等于非线性比历史基线好。

举一个抽象例子：

```text
线性单因素比历史基线差 10%
非线性单因素比历史基线差 3%
```

这时 `nonlinear_minus_linear` 是正的，因为非线性比线性好；但非线性仍然没有超过历史基线，所以不能说这个因素已经有预测增益。

这就是为什么 D2.6 需要证据桶。

### 5.10 D2.6 的证据桶怎么读

脚本用 `evidence_bucket` 给每个组合分类：

```python
if nonlinear_imp >= 0.05 and linear_imp < 0.05 and delta >= 0.03:
    return "linear_weak_but_nonlinear_improves"
if linear_imp >= 0.05 and nonlinear_imp >= 0.05 and delta >= 0.03:
    return "linear_effective_nonlinear_extra_gain"
if linear_imp >= 0.05 and delta < 0.03:
    return "linear_already_effective_no_clear_nonlinear_gain"
if "nonlinear" in d1_verdict and nonlinear_imp < 0:
    return "d1_shape_line_but_no_prediction_gain"
if nonlinear_imp >= 0 and delta > 0:
    return "nonlinear_slight_line"
return "no_nonlinear_prediction_gain"
```

逐个翻译：

| 证据桶 | 中文理解 | 该怎么用 |
| --- | --- | --- |
| `linear_weak_but_nonlinear_improves` | 线性模型不强，但非线性模型超过历史基线且明显优于线性 | 最值得复核，说明线性可能低估了这个因素 |
| `linear_effective_nonlinear_extra_gain` | 线性本来就有效，非线性还能再提升 | 也是好候选，但需要判断非线性是否稳定 |
| `linear_already_effective_no_clear_nonlinear_gain` | 线性已经有效，非线性没明显额外提升 | 优先用线性解释，不必复杂化 |
| `d1_shape_line_but_no_prediction_gain` | D1.5 有形态线索，但非线性回测没有超过历史基线 | 不能进入预测主线，可保留为解释线索 |
| `nonlinear_slight_line` | 非线性略有线索，但强度不够 | 谨慎观察 |
| `no_nonlinear_prediction_gain` | 非线性没有预测增益 | 暂不保留 |
| `too_short_or_unstable` | 样本或结果不足 | 不判断 |

这张表比单看图更重要。读 D2.6 图时，你不能只看横条在 0 线右边还是左边，还要看颜色代表的证据桶。

### 5.11 D2.6 的总体实验结果

D2.6 完成 230 个字段级组合。按证据桶统计：

| 证据桶 | 数量 | 说明 |
| --- | ---: | --- |
| `d1_shape_line_but_no_prediction_gain` | 134 | D1.5 有形态线索，但没有转化为严格滞后预测增益 |
| `nonlinear_slight_line` | 26 | 非线性略优于线性，但证据强度不足 |
| `linear_already_effective_no_clear_nonlinear_gain` | 25 | 线性已能解释主要增益，非线性不必优先 |
| `linear_weak_but_nonlinear_improves` | 21 | 线性弱但非线性改善，后续重点复核 |
| `no_nonlinear_prediction_gain` | 12 | 非线性没有形成预测增益 |
| `linear_effective_nonlinear_extra_gain` | 10 | 线性有效，非线性仍有额外增益 |
| `too_short_or_unstable` | 2 | 样本或结果不足 |

最重要的读法：

- 134 个组合属于“有形态但无预测增益”，说明 D1.5 的非线性形状不能直接写成预测结论。
- 21 个组合属于“线性弱但非线性改善”，说明你提出的担心是成立的：确实存在一些因素如果只用线性 Ridge 会被低估。
- 10 个组合属于“线性有效且非线性额外改善”，说明有些因素不仅线性有效，还可能存在非线性增益。
- 但 21 + 10 加起来也只有 31 个组合，所以非线性不是万能解决方案，而是帮我们更细地筛候选。

按一级分类看：

- 天气字段的非线性改善线索最多，尤其在第一产业、第二产业上更明显。
- 经济字段也有若干逐字段线索，但必须逐字段解释，不能写成“经济因素整体有效”。
- 新型负荷只有少量线索，仍是谨慎候选。
- 内部业务字段没有形成突出的非线性增益主线。

### 5.12 图 34：城乡居民生活用电合计 D2.6 非线性相对线性差值

![图34](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_6_single_factor_nonlinear_evidence/matplotlib_charts/001_nonlinear_minus_linear_B_城乡居民生活用电合计.png)

这张图的横轴是：

```text
最佳非线性模型的 WAPE 改善率 - 线性单因素 Ridge 的 WAPE 改善率
```

右侧表示非线性比线性好，左侧表示非线性比线性差。

但请注意颜色。城乡居民生活用电合计这里没有 `linear_weak_but_nonlinear_improves`，也就是没有出现“线性弱但非线性真正改善预测”的强候选。

这张图里的很多右侧横条仍然属于 `d1_shape_line_but_no_prediction_gain`。这意味着：

- 非线性可能比线性单因素好一点。
- 但它仍然没有超过历史基线。
- 所以不能说这个因素已经带来预测增益。

当前说明：

- 居民用电在 D1.5 中确实有明显非线性形状，尤其温度相关因素很符合“夏冬双峰”的直觉。
- 但在 D2.6 的严格滞后预测验证中，这些形状没有稳定转化为预测误差改善。
- 可能原因包括：居民用电更依赖同月实际气温或天气预报，而本轮 D2.6 只能用天气观测滞后项；历史目标和月份周期已经吸收了很多季节结构。

当前不能说明：

- 不能说居民用电没有非线性。
- 也不能说天气对居民用电没影响。
- 只能说：在当前滞后口径、当前样本和当前低成本非线性模型下，居民层级没有形成强预测增益候选。

### 5.13 图 35：全社会用电总计 D2.6 非线性相对线性差值

![图35](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_6_single_factor_nonlinear_evidence/matplotlib_charts/002_nonlinear_minus_linear_全社会用电总计.png)

全社会用电总计里出现了少量强候选：

- `linear_weak_but_nonlinear_improves` 有 2 个组合。
- `linear_effective_nonlinear_extra_gain` 有 2 个组合。
- 但 `d1_shape_line_but_no_prediction_gain` 仍然有 34 个组合。

这说明总量层级的非线性增益不是主导现象。多数 D1.5 的形态线索，在总量预测中仍然没有超过历史基线。

怎么读这张图：

1. 先看最靠右的条，说明非线性比线性更好。
2. 再看颜色，如果颜色对应 `d1_shape_line_but_no_prediction_gain`，就不能当作预测增益。
3. 只有进入 `linear_weak_but_nonlinear_improves` 或 `linear_effective_nonlinear_extra_gain` 的字段，才值得作为下一轮候选。

当前说明：

- 总量层级历史基线较强，非线性模型不容易显著超过。
- 有少量天气和价格类字段出现非线性线索，可以保留为候选。
- 但总量层级不应把 D2.6 写成“非线性模型整体显著提升预测”。

当前不能说明：

- 不能说非线性模型对总量整体有效。
- 不能说经济或天气作为整体类别有效。
- 只能说：总量层级存在少量逐字段非线性预测增益线索。

### 5.14 图 36：第一产业 D2.6 非线性相对线性差值

![图36](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_6_single_factor_nonlinear_evidence/matplotlib_charts/003_nonlinear_minus_linear_第一产业.png)

第一产业是 D2.6 里比较值得看的目标层级。

证据桶统计显示：

- `linear_weak_but_nonlinear_improves` 有 3 个组合。
- `linear_effective_nonlinear_extra_gain` 有 4 个组合。
- `nonlinear_slight_line` 有 7 个组合。

候选主要集中在天气字段，例如：

- 短波辐射。
- 降水时长。
- 降雨量。
- 降水量。
- 平均最低气温。

这些字段为什么可能是非线性的？

第一产业用电与天气的关系通常不是“天气指标越高，用电越高”这么简单。比如：

- 适度降水可能降低灌溉需求，但极端降水又可能影响生产活动。
- 辐射、气温、降水共同影响农业生产节奏。
- 低温、高温、降雨这些因素可能存在阈值，而不是线性变化。

D2.6 的结果支持一个谨慎判断：

> 第一产业中，天气滞后因素不仅有线性预测线索，也存在低成本非线性模型进一步改善的候选。

但边界也很明确：

- 这不是因果证明。
- 不是说所有天气字段都有效。
- 不代表可以直接上线非线性模型。
- 还需要异常月份、极端天气事件和农业生产节奏解释来复核。

### 5.15 图 37：第三产业 D2.6 非线性相对线性差值

![图37](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_6_single_factor_nonlinear_evidence/matplotlib_charts/004_nonlinear_minus_linear_第三产业.png)

第三产业的结果介于总量和第二产业之间。

证据桶统计显示：

- `linear_weak_but_nonlinear_improves` 有 2 个组合。
- `linear_effective_nonlinear_extra_gain` 有 2 个组合。
- `linear_already_effective_no_clear_nonlinear_gain` 有 5 个组合。
- `d1_shape_line_but_no_prediction_gain` 有 32 个组合。

这说明第三产业有一些非线性候选，但大部分 D1.5 形态线索仍然没有转化成严格预测增益。

当前靠前的候选包括：

- 社会消费品零售总额绝对量。
- 进出口总额绝对量。
- 平均体感温度。
- 平均最低气温。

怎么理解：

- 第三产业包含服务业、商业、交通、信息等多种活动，经济景气和天气舒适度都可能影响用电。
- 但这些机制并不统一，所以不能把第三产业结果写成“经济因素整体有效”或“天气因素整体有效”。
- 更合适的说法是：第三产业存在少量逐字段非线性候选，值得后续按业务机制拆开复核。

当前不能说明：

- 不能把第三产业所有经济字段打包成一类结论。
- 不能说非线性模型在第三产业稳定优于线性模型。
- 不能替代正式行业层级分析。

### 5.16 图 38：第二产业 D2.6 非线性相对线性差值

![图38](/Users/neonho/Documents/GitHub/js-power-forecast/output/modeling/mvp/d2_6_single_factor_nonlinear_evidence/matplotlib_charts/005_nonlinear_minus_linear_第二产业.png)

第二产业是 D2.6 中非线性线索最集中的目标层级。

证据桶统计显示：

- `linear_weak_but_nonlinear_improves` 有 14 个组合。
- `linear_effective_nonlinear_extra_gain` 有 1 个组合。
- `nonlinear_slight_line` 有 11 个组合。
- `d1_shape_line_but_no_prediction_gain` 只有 13 个组合，相比其他目标层级少很多。

这意味着：第二产业里，确实有一批因素在线性 Ridge 下表现不强，但换成浅层树、二次项 Ridge 或浅随机森林后，预测误差改善了。

候选主要包括：

- 降雪量。
- 平均相对湿度。
- 降水量、降雨量。
- 平均云量。
- 平均气温、最低气温、体感温度。
- 部分 CPI、外贸、社零、工业增加值字段。
- 工商业充电设施电量。

怎么理解：

第二产业用电与天气、经济景气、生产节奏之间可能存在阈值或分段关系。例如：

- 某些天气条件在轻微变化时影响不大，但超过阈值后影响生产或用能。
- 外贸和价格指标可能在景气较弱或较强区间对生产用电有不同影响。
- 新型负荷和工业生产活动的关系可能不是线性的。

但这张图也最需要谨慎：

- 第二产业里不少最佳模式是 `sklearn_decision_tree_depth2`。
- 浅层树可以捕捉阈值，但月度样本短，阈值可能受个别月份影响。
- 因此这些字段应该进入下一轮稳定性复核，而不是直接写成稳定规律。

当前可以支持的谨慎结论：

> 第二产业是当前最值得继续做逐字段非线性复核的层级，尤其是天气类字段和少量经济/新型负荷字段。

当前不能支持的结论：

- 不能说第二产业非线性模型已经可以上线。
- 不能说所有天气因素都对第二产业有效。
- 不能说 CPI、外贸、社零等经济因素属于同一个机制。
- 不能忽略异常月份、口径变化和重大生产事件。

### 5.17 D2.6 和 D1.5 / D2.5 的关系应该怎么讲

现在三层证据可以合起来理解：

1. D1.5 发现了很多非线性形态线索。
2. D2.5 发现多数因素用线性单因素滞后项并不能改善历史基线。
3. D2.6 进一步发现：确实有一部分因素被线性模型低估，尤其在第二产业和第一产业天气字段上更明显。

但 D2.6 同时也提醒我们：

1. 大量 D1.5 形态线索没有转化为预测增益。
2. 非线性模型不是越复杂越好。
3. 短月度样本下，浅层非线性模型只能作为候选筛选工具。
4. 真正进入阶段成果稿时，只能写“部分因素存在非线性预测增益线索”，不能写成因果或上线结论。

一句最稳妥的阶段表述是：

> D2.6 逐因素非线性回测显示，少量字段在低成本非线性模型下相对线性单因素模型出现额外预测改善，第二产业和第一产业天气相关字段更值得后续复核；但多数 D1.5 非线性形态线索尚未转化为严格滞后预测增益，当前仍应作为候选证据而非最终结论。

## 6. 你现在可以怎样复核这些结果

如果你想从代码核对结果，建议按这个顺序看：

1. 先看 D2.5 的 `add_prediction_time_features`：确认因素同月值没有进入预测。
2. 再看 `single_factor_spec`：确认每次只加入一个因素的滞后项。
3. 再看 `prepare_design_matrices`：确认标准化只使用训练集均值和标准差，测试集没有泄漏。
4. 再看 `ridge_predict`：确认 Ridge 是 `numpy` 闭式解，不是神经网络。
5. 再看 D2.6 的 `MODEL_SPECS`：确认它比较了历史基线、线性单因素、二次项 Ridge、浅层树和浅随机森林。
6. 再看 D2.6 的 `build_poly2_ridge`：确认二次项只作用于当前因素的滞后项，而不是把全部历史目标特征都平方。
7. 再看 D2.6 的 `evidence_bucket`：确认“非线性比线性好”和“非线性比历史基线好”没有被混为一谈。
8. 最后看各脚本的 `build_charts`：确认图上的横条分别来自 `wape_improvement_vs_history` 或 `nonlinear_minus_linear`。

如果你想从结果核对图，建议按这个顺序看：

1. D1.5：先找非单调图，判断线性模型可能漏掉什么。
2. D2.5：看每个目标层级右侧因素有多少，判断哪些因素真的带来预测改善。
3. D2.6：看非线性相对线性是否为正，再看证据桶是否真正超过历史基线。
4. calendar_holiday：单独看，因为它是预测时点已知因素，不必和同月经济/天气观测混在一起。
5. D3.5：看每个行业最靠右的因素是否一致。如果不一致，就说明不能把总量结论直接搬到行业。

## 7. 当前最重要的结论边界

1. D1.5 说明：线性筛选会漏掉一些非线性候选，特别是天气、日历和部分经济字段。
2. D2.5 说明：逐因素验证后，多数因素并没有改善历史基线，因此不能只凭理论相关进入预测主线。
3. D2.6 说明：确实存在一部分“线性弱但非线性改善”的字段，尤其第二产业和第一产业天气字段值得复核；但多数 D1.5 形态线索没有转化为预测增益。
4. calendar_holiday 说明：结构化日历因素是当前比较干净的预测时点已知变量，对总量和产业层级有轻量验证支持。
5. D3.5 说明：试点行业之间差异明显，行业层级不能直接套用总量或产业结论。
6. 这些图都不是因果证明，也不是最终模型上线证明；它们的价值是帮我们筛出下一轮值得认真建模和解释的因素。
