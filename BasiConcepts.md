# Recommendation Systems Learning
# Part 1: Overview
召回 -> 粗排 -> 精排 -> 重排

## 1. 召回 （Candidate Generation）
	•	目标：从海量物品中快速筛选出一个规模较小但相关性较高的候选集。
	•	特点：
	•	对效率要求极高，通常用轻量级模型或规则。
	•	常用方法：协同过滤、Embedding 向量相似度、用户-物品共现矩阵、热度排序、用户行为规则。
	•	规模：从几十万到几百万条候选集压缩到几千条

已经小红书为例，一般召回阶段使用很多召回通道得到几千个笔记，然后做去重和过滤（过滤掉用户不感兴趣都作者和内容等）

## 2. 粗排（Pre-Ranking）
 主要用户小型机器学习对笔记打分，根据得分进行排序。这阶段笔记很多，所以小红书模型效率更快。

	•	目标：进一步过滤候选集，初步排序。
	•	特点：
	•	模型复杂度适中，兼顾效率和效果。
	•	通常使用轻量级机器学习模型（如LR、GBDT等）。
	•	特征主要来源于用户基本属性和行为数据的统计特征。
	•	规模：从几千条候选集筛选到几百条。

## 3. 精排（Ranking）
使用大型神经网络继续评分排序

	•	目标：对粗排结果进行精细排序，以提高推荐效果。
	•	特点：
	•	使用复杂模型，关注用户兴趣的精准匹配。
	•	常用方法：基于深度学习的排序模型（如DNN、Wide&Deep、DIN、Transformer等）。
	•	特征丰富，可能包括用户个性化特征、物品特征、上下文特征、交叉特征等。
	•	规模：从几百条候选集最终筛选出几十条。

## 4. 重排（Re-Ranking）
主要考虑多样性，对精排结果随机抽样，并按规则打乱，然后插入广告等内容

	•	目标：在精排结果的基础上进一步优化，兼顾业务需求。
	•	特点：
	•	考虑多目标平衡（如点击率、转化率、覆盖率、多样性、时效性等）。
	•	融合业务规则（如去重、特定物品曝光限制等）。
	•	可能使用多目标优化模型或规则调整。
	•	规模：最终选择几条或几十条进行推荐展示。

常用MMR和DDP等算法进行多样性抽样，然后根据规则打散相似的样本，

一般粗排精排的模型会使用用户特征，物品特征等作为输入，输出则是模型预测的点击率，点赞率，收藏率等指标，然后需要融合这些指标进行最终分数评估。

## 5. 推荐系统 ab test
一般先离线实验，如果反馈正向----
小流量线上ab test----
全量上线

随机分流部分可以用户哈希函数对用户id进行映射（可以认为哈希映射结果是接近随机的），比如
hash（id）=id%3，保证id接近的用户经过映射后的整数耶会相差很多。
然后将哈希后的整数进行分桶，比如100个用户分10桶，每个桶10用户

流量不足的处理：分层实验
比如不同阶段分为不同层（召回蹭，粗排层等），同层的实验互斥，不同层正交。一个用户不能同时受两个召回实验影响，但可以同时召回和粗排实验。

一般同类策略（如精排的两种结构）天然互斥，或两个类型相同的召回实验可能会互相增强或抵消，对于一个用户只能用其中一种。而不同层一般不会互相干扰，可以进行正交（同时参与召回和粗排实验）

![figures/fig1.png](figures/fig1.png)

## 6. 反转实验
一些指标（点击，交互等）可以立刻收到新策略影响，而另一些比如留存率等，存在滞后性。

反转实验是指：在新的推全层上保留一个小的反转桶，使用旧策略，长期观察新旧策略的diff来评估某些有滞后性的指标变化


## 7. 冷启动：Cold Start
**冷启动（Cold Start）**是推荐系统中一个常见的问题，指的是在用户或物品缺乏足够历史数据的情况下，推荐系统难以有效生成个性化推荐的情况。冷启动问题通常发生在以下几种场景中：

---

### **1. 冷启动的类型**

#### **（1）用户冷启动**
- **场景**：新用户刚注册，没有任何交互行为（如浏览、点击、购买等），系统无法准确了解其兴趣和偏好。
- **影响**：系统无法为新用户生成个性化推荐，容易推荐热门物品或随机推荐，导致推荐精准度较低。

#### **（2）物品冷启动**
- **场景**：新物品刚上线，没有任何用户与其交互记录。
- **影响**：系统无法将新物品推荐给合适的用户，可能导致新物品曝光不足或滞销。

#### **（3）系统冷启动**
- **场景**：一个新上线的推荐系统缺乏足够的用户和物品交互数据。
- **影响**：系统难以构建有效的推荐模型。

---

---

### **2. 冷启动问题的常见挑战**

1. **数据不足**：
   - 缺乏用户和物品交互数据，推荐模型难以训练或生成有效嵌入。

2. **推荐精准度低**：
   - 冷启动阶段通常依赖规则或热门推荐，个性化效果较差。

3. **用户流失风险**：
   - 冷启动阶段推荐质量低，可能导致用户流失。

4. **多样性和公平性问题**：
   - 新物品可能缺乏曝光机会，推荐结果容易集中在已有热门物品上。

---

### **3. 冷启动的解决策略总结**

| **冷启动类型**    | **方法**                                                                                   | **优点**                               | **缺点**                                   |
|------------------|------------------------------------------------------------------------------------------|----------------------------------------|-------------------------------------------|
| **用户冷启动**    | 用户画像、兴趣问卷、热门推荐、基于相似用户的推荐                                            | 快速了解新用户兴趣                     | 个性化不足，初期推荐效果可能较差            |
| **物品冷启动**    | 基于内容推荐、规则驱动、相似物品协同过滤、探索性推荐                                        | 快速曝光新物品                         | 曝光效果依赖物品特征的质量                 |
| **系统冷启动**    | 引入外部数据、多策略结合、探索性模型                                                       | 有助于快速收集数据                     | 难以平衡探索与推荐质量                     |

---




# Part 2: Recall
召回是推荐系统第一个阶段，用于从海量物品中快速筛选出一个规模较小但相关性较高的候选集。召回阶段的目标是尽可能多地覆盖用户的兴趣，同时保证召回的准确性和效率。


以电商推荐系统为例，可能的召回处理流程如下：

1. **多通道召回**：
   - **协同过滤**：从用户历史购买记录召回相似物品（Item CF）。
   - **双塔模型**：结合用户画像和物品多模态特征，生成个性化召回。
   - **热门推荐**：补充近期销量或浏览量高的热门物品。
   - **新品召回**：加入近期上架的新品。

2. **候选合并**：
   - 合并各通道结果，赋予不同通道权重（如双塔模型权重最高）。

3. **去重与过滤**：
   - 去除重复物品，过滤掉已购买或不符合推荐条件的物品。

4. **排序优化**：
   - 使用排序模型综合考虑用户兴趣、物品特征和业务规则，生成最终推荐列表。

---

### **总结**
- **多通道召回**是推荐系统提升覆盖率和多样性的关键策略，不同通道方法侧重于不同场景和问题。
- 合并召回结果时需要考虑通道权重、去重和排序优化，以确保推荐结果既精准又多样。
- 最终的召回结果为后续的排序阶段提供高质量的候选集，从而优化用户体验和业务目标。

## 1. 协同过滤（Collaborative Filtering）
    •	思想：根据用户的历史行为，找到与用户历史行为相似的物品，将这些物品推荐给用户。
    •	优点：简单、易于实现。
    •	缺点：无法利用用户的个性化信息，容易出现热门物品的推荐。
    •	算法：主要包括基于物品的协同过滤算法、基于用户的协同过滤算法、基于模型的协同过滤算法等。

### 1.1 Item Based CF
每个用户交互过很多item，如果用户喜欢item1，并且item1和item2相似，那么用户可能也会喜欢item2。

该方法首先需要计算item之间的相似度，然后预估用户对未交互item的评分，最后根据评分进行推荐。

#### User-Item Matrix
|        | Item1 | Item2 | Item3 | Item4 | Item5 |
|--------|-------|-------|-------|-------|-------|
| User1  | 5     | 3     | 0     | 0     | 2     |
| User2  | 2     | 0     | 0     | 1     | 4     |
| User3  | 0     | 0     | 4     | 3     | 0     |

该矩阵表示不同用户对不同item的评分（兴趣），比如click算1分，like算2分，share算3分等，计算每个用户对每个item的评分，得到矩阵。

#### Item-Item Similarity
计算item之间的相似度，一般使用余弦相似度等方法，得到item之间的相似度矩阵。

|        | Item1 | Item2 | Item3 | Item4 | Item5 |
|--------|-------|-------|-------|-------|-------|
| Item1  | 1     | 0.8   | 0.2   | 0.4   | 0.6   |
| Item2  | 0.8   | 1     | 0.3   | 0.5   | 0.7   |

通过该矩阵，输入itemID，可以查找出k个最相似的item，然后根据用户对这些item的评分，预测用户对该item的评分。

#### 线上召回：
- 给定用户id，通过User-Item Matrix查找用户交互过的item（n个），
- 然后通过Item-Item Similarity查找每个item的k个最相似item（k个），
- 对于取回的item（n*k个），预估用户对这些item的评分，排序后取topN作为召回结果。

通过这两个矩阵的索引可以避免枚举所有item，离线计算量大，但离线计算完成后储存结果，线上查询效率高。

### 1.2 Swing 召回
item Based CF存在问题：当与两个item交互的用户存在一个小圈子中，那么很可能这两个item相似度很高，
但是实际上可能是因为小圈子用户重叠导致的，而不是因为item本身相似。

如果大量不相关用户交互两个item，说明这两个item可能有相同受众。

Swing模型就是通过给用户设置权重，通过在相似度计算公式添加用户重合比例权重，若两个item重合用户比例高，则分母上该权重变大，
总相似度降低，从而避免小圈子效应。


### 1.3 User Based CF
与Item Based CF类似，计算用户之间的相似度，然后根据已知用户对item的评分，预测新用户对item的评分。

- 该算法考虑的一个点：热门item对相似度的影响，如果一个item很热门，那么很多用户都会交互，该item实际对用户相似度计算的价值不高。因此需要对热门item进行权重调整。
- 冷门item更能反映出用户相似度，若两个user都对一个冷门item有交互，那么这两个user的相似度更高。
- 总结：热门item降低权重后计算user similarity，然后根据user similarity预测用户对item的评分。

整体流程和Item CF类似：
- User-Item Matrix：用户对item的交互
- User-User Similarity：用户之间的相似度

线上召回：
- 给定用户id，通过User-User Similarity查找用户相似度最高的k个用户，
- 通过User-Item Matrix查找这k个用户近期交互的item（last-n）
- 对于取回的item，预估用户对这些item的评分，排序后取topN作为召回结果。


## 2. 双塔模型

![figures/fig2.png](figures/fig4.jpg)

双塔模型是一种基于Embedding的召回模型，通过Embedding向量表示用户和item，然后计算用户和item的相似度，最后根据相似度进行召回。
该方法是传统协同过滤的一种改进，工业界常用。

**双塔模型**(Two-Tower Model)是一种常用于推荐系统和信息检索中的深度学习架构，旨在高效处理大规模用户与物品（User-Item）匹配问题。它通过分别为用户和物品构建独立的特征表示向量，
并在匹配阶段计算二者的相似性（例如内积或余弦相似度）作为**兴趣评分的预测值**，从而完成推荐或检索任务。

---

### **双塔模型的结构**
双塔模型通常由两部分组成：用户塔（User Tower）和物品塔（Item Tower）。它们的作用如下：

1. **用户塔（User Tower）**
   - 用于提取用户特征的表示向量（Embedding）。
   - 输入数据包括用户ID、历史行为（如点击、购买记录）、上下文特征（如时间、地理位置）等。
   - 通过嵌入层（Embedding Layer）和若干全连接层（或其他神经网络结构）生成用户特征向量。

2. **物品塔（Item Tower）**
   - 用于提取物品特征的表示向量（Embedding）。
   - 输入数据包括物品ID、属性特征（如品类、价格、描述）等。
   - 同样通过嵌入层和若干全连接层生成物品特征向量。

3. **相似性计算**
   - 用户塔和物品塔输出的向量通常在一个共享的嵌入空间中。
   - 通过计算用户向量和物品向量的相似度（如内积或余弦相似度），衡量二者的匹配程度。

4. **优化目标**
   - 使用负采样策略构建正负样本对，采用目标函数（如二分类交叉熵或最大化相似度）进行训练。

---

### **双塔模型的优势**
1. **高效性**
   - 双塔模型解耦了用户和物品的特征提取，计算复杂度低。
   - 在离线阶段，物品向量可以提前计算并索引，仅需在线计算用户向量和物品的相似性即可完成推荐。

2. **扩展性**
   - 支持海量用户和物品的推荐任务。
   - 特别适用于检索场景中需要快速匹配的情况（如召回阶段）。

3. **灵活性**
   - 用户塔和物品塔可以独立设计，支持多样化的输入特征和网络结构。

---
    
### **双塔模型的不足**
1. **特征独立性限制**
   - 用户塔和物品塔独立训练，难以捕捉用户和物品之间复杂的交互关系。
   - 解决方案：可以在召回后通过排序模型补充交互信息。

2. **冷启动问题**
   - 对于缺乏历史行为的新用户或新物品，双塔模型可能无法生成准确的向量。
   - 解决方案：引入基于内容的特征（如用户画像、物品属性）缓解冷启动问题。

3. **潜在表达能力不足**
   - 双塔模型假设用户和物品可以通过简单的相似性度量匹配，可能无法满足某些复杂场景的需求。

双塔模型主要通过对用户和item分别进行多特征维度的嵌入（分类特征进行embedding，连续变量进行归一化或分桶等处理），使用神经网络进行特征提取，最后得到每个用户和item的嵌入向量，再分别计算
User-Item的相似度，最后根据相似度进行召回。

### 双塔模型的训练


| 方法       | **样本处理方式**                              | **正负样本关系**                                        |
|------------|-----------------------------------------|--------------------------------------------------------|
| **Pointwise** | 独立处理每个样本，将正样本和负样本看作独立的二分类或回归任务。         | 无需比较，正负样本之间无直接关系。                       |
| **Pairwise**  | 每次取一个用户，一个正样本物品和一个负样本物品，通过构造正负样本对来优化模型。 | 强调正负样本对的相对关系，目标是正样本得分高于负样本。     |
| **Listwise**  | 每次取一个用户，一个正样本物品和多个负样本物品                 | 同时考虑多个正负样本之间的全局排序关系，直接优化排名指标。 |

---
####  **总结**
- **Pointwise** 独立看待正负样本，适合简单的二分类或回归问题。
- **Pairwise** 强调正负样本对之间的相对关系，适合优化排序。
- **Listwise** 同时处理多个正负样本，从全局优化排序效果，适合需要整体排名的场景。

![figures/fig2.png](figures/fig2.jpg)

对于PairWise，基本思想是鼓励cos（a，b+）>cos（a，b-），
损失函数为：
- Triplet Hinge Loss： max(0, margin + cos(a, b-) - cos(a, b+))
  - margin为超参数，一般取1
- Triplet Log Loss： log(1 + exp(cos(a, b-) - cos(a, b+)))


## 3. 召回模型训练时的正负样本选择
![figures/fig2.png](figures/fig3.jpg)

### 3.1 简单负样本
指的是未被召回的物品，大概率是用户不感兴趣都，约等于全体物品（大部分都不会被召回），因此：
- 直接对全体物品进行抽样，作为负样本。
- 由于正样品大部分为热门物品，如果均匀抽样会导致产生的负样本大多是冷门物品，因此需要非均匀抽样（减少热门物品对冷门物品的影响）
- 非均匀抽样：负样本抽样概率应该与热门程度（如点击次数）正相关---
  - 抽样概率 = （点击次数）^α，α为超参数，一般取0.75
  - 点击次数越高，抽样概率越大，但不是线性关系

### 3.2 困难负样本
指的是排序（粗排，精排）阶段被淘汰的样本。这样的样本已经被召回了，即用户有一定的兴趣，只是被排序截断。这样的样本容易被误分类为正样本。
- 工业界一般进行负样本混合：用简单负样本和困难负样本混合作为负样本用于训练

#### 负样本的错误选择：
对于召回模型，只能使用上述的easy/hard负样本（源于召回和排序阶段），但不能使用已经曝光但没有点击的样本作为负样本。
- 已经曝光但样本表示该样本已经经过一些列链路在重排后展示给用户，可能用户只是恰好没有点击；
- 召回模型目的是区分用户感兴趣或不感兴趣，而不是区分用户感兴趣或更感兴趣；
- 曝光的样本是我们认为用户感兴趣的部分，应该属于正样本。**这些样本可以用于排序模型作为负样本。**

## 4. 双塔模型线上召回和更新

- 训练好双塔模型后，离线储存item塔的向量（向量数据库），即每个item的特征向量b
  - 对item向量库建立索引，便于加速最近邻查找
- 线上召回：
  - 用户发起推荐请求
  - 线上神经网络根据用户id和画像（特征），实时计算用户向量a
  - 使用用户向量a作为query，通过item向量库查找最相似的k个item
  - 这些item会和CF，swing等通道进行融合，作为最终召回结果

### 为什么用户向量实时计算，而item向量离线储存？
- 每次召回只需要计算一个用户向量，而需要从几亿个item中进行召回
- 计算一个用户向量a是可接受的实时计算，而item向量过大，只能采取离线存储并查找的手段。
- 而实时计算用户向量可以根据用户兴趣变化而动态推荐，item的向量相对稳定，不需要频繁调整。

### 全量更新
- 每天凌晨使用昨天的数据重新训练双塔模型，
- 在昨天的参数上继续训练（不是随机初始化），训练一个epoch一般，
- 发布新的用户塔神经网络以及更新item向量库。

### 增量更新
online learning更新模型参数，让模型在用户交互后几小时内就能反映用户兴趣变化
- 实时收集线上数据，做流式处理，生成TFRecord文件；
- 对模型做online learning，增量更新ID Embedding参数（不更新其他参数）
- 即从早到晚不断生成训练数据，不断进行梯度下降更新embedding层，冻结其余全连接层
- 发布用户ID Embedding，供用户塔神经网络实时计算用户向量。

### 能否只做增量更新而不做全量？
- 如果只做增量更新，由于小时间段内数据是有偏的（用户的兴趣变化大），可能会导致模型过拟合，因此需要全量更新来平衡模型的泛化能力和精确度。
- 全量更新时需要random shuffle全天数据，做1 epoch训练；而增量更新可以看作是时间序列数据，并没有打乱。


## 5. Deep Retrieval
区别于双塔模型（向量召回，将用户和item都嵌入为向量后进行query），Deep Retrieval基于路径进行召回

## 6. 其他召回通道
### 1. 地理位置召回
- GeoHash召回：思想是用户可能对附近发生的事感兴趣，对经纬度编码（二进制哈希码），geohash作为key，item作为value
  - 该方法召回只关注用户自身地理位置，并没有个性化
- 同城召回：召回同一城市的item信息

### 2. 作者召回
- 用户对某个作者感兴趣（关注该作者），索引为：
  - 用户--作者
  - 作者--item（按时间倒排，最新的排在最前面）
  -  召回形式：用户-->作者-->item

- 也可以扩展为“有交互的作者召回（点赞，收藏等）”
- 相似作者召回：
  - 作者--相似作者
  - 用户-->感兴趣的作者-->相似作者（相似度）-->item

### 3. 缓存召回
复用之前n次推荐过程中精排的结果。由于精排结果会进行重排后多样性抽样，最终给用户的item只是精排结果的一部分；
精排中还有很多item并没有展示，可以将这些item作为缓存召回的候选集。
- 由于缓存大小固定，需要退场机制
- 比如一旦笔记曝光成功，需要从缓存中删除
- 如果缓存大小达到上限，先移除最先进入缓存的item

## 7.曝光过滤--Bloom Filter
需要对已曝光的item进行记录，保证下一次重拍后推荐给用户的item是没被曝光过的。一般使用Bloom Filter进行过滤。

可以设置时间限制，比如超出一个月的item就从物品集移除，减少误删未曝光物品的概率。

![figures/fig2.png](figures/fig5.jpg)


# Part 3: Ranking
Ranking 包括：粗排、精排、重排；促排和精排原理相似，都是使用模型对item进行打分后排序，只是粗排模型要简单，计算效率高，需要从大量recall的结果中筛选截断一部分；
而精排模型更复杂，需要更多的特征，更多的计算，但是效果更好。

## 1. Multi-task Learning：多任务学习
用户-物品交互数据是推荐系统中的重要信息，包括用户的点击、购买、收藏、点赞等行为。这些交互数据反映了用户对物品的兴趣和偏好，是推荐系统训练模型的重要数据来源。
- impression：曝光，用户看到了该物品
- clicks: 用户点击物品的行为（点击率=点击次数/曝光次数）
- likes：用户点赞物品的行为（点赞率=点赞次数/点击次数）
- collect：用户收藏物品的行为（收藏率=收藏次数/点击次数）
- shares：用户分享物品的行为（分享率=分享次数/点击次数）

### 排序的依据
使用Ranking Model预估点击率（CTR），点赞率等指标分数，通过加权和等方式融合分数后对分数进行排序，截断topN作为推荐结果。

### 精排模型核心：shared bottom
![figures/fig2.png](figures/fig6.jpg)

一般如上图所示：
- 将不同特征进行concat，然后使用一个神经网络（可以是全连接或wide&deep或其他结构），得到一个嵌入表征向量
- 将该向量分别输入四个不同的下游模型，分别预测点击率，点赞率，收藏率，分享率等指标
- 最后将这四个指标融合，得到最终的排序分数

上述模型的优点：
- shared bottom：模型大，需要对融合后的特征进行表征
- 属于**前期融合**：先对所有特征进行concatenation再输入神经网络
- 线上推理代价大：给n个item打分需要推理n次

#### 双塔模型属于后期融合：先对不同特征输入不同神经网络，不直接融合特征。优点是线上推理计算量小，用户塔只需要计算一次用户表征a；
#### 物品表征b可以离线推理后储存在向量数据库，线上推理的时候直接使用

### 排序模型的训练

![figures/fig2.png](figures/fig7.jpg)

yi是用户的行为（0或1），pi是子网络的预测值（0-1之间）。对于每个子网络相当于做二分类。

面对的困难：不平衡。一般常用对**负样本下采样**。
- 原始正负样本数为n+,n-
- 负样本下采样，使用采样率α，则采样后负样本数为αn-
- 由于负样本变少，**预估的指标（点击率）会大于真实点击率，需要进行校准**

![figures/fig2.png](figures/fig8.jpg)



### **1. 预估值校准的作用**
预估值校准的目标是让模型的输出分数更接近实际概率（真实点击率 \(p_{\text{true}}\)），而不仅仅是一个用来排序的相对分数。这在以下场景中可能是必要的：

#### **（1）需要解释性**
校准后的分数可以更直观地解释。例如，如果校准后的预测值是 0.7，那么可以解释为“该用户点击的概率为 70%”。这对业务分析或用户反馈很有帮助。

#### **（2）跨任务或跨模型的一致性**
- 如果你的系统有多个目标（如点击率、购买率等），校准分数可以确保不同任务的评分在同一范围内（比如概率范围 [0, 1]）。
- 在多个模型融合（如排序模型与过滤模型结合）时，校准分数可以统一不同模型的分数尺度，避免因分数范围不同引入偏差。

#### **（3）对决策产生影响**
对于某些推荐系统，不仅仅需要排序结果，还需要使用分数来调整推荐策略。例如：
- 在 **多目标优化** 中，可能需要根据校准后的概率分数加权不同目标的重要性。
- 在 **广告竞价系统** 中，校准后的点击率直接参与收益计算（例如计算预期收益）。

---

### **2. 校准对排序的影响**
从排序模型的角度，如果仅仅关心排序结果，相对大小比绝对值更重要，因此校准并不总是必要。但有以下情况需要考虑：

#### **（1）截断点的位置选择**
排序模型通常需要预测所有候选物品的分数后，选择前 \(k\) 个结果。如果分数未校准且范围偏移严重，可能会影响截断点附近物品的排序质量。例如：
- 未校准的分数可能导致前 \(k\) 个物品和第 \(k+1\) 个物品之间的分数差异不具有实际意义。

#### **（2）样本偏置问题**
排序模型训练时往往使用点击数据，但点击数据通常存在样本偏置（例如展示过的内容被点击的概率远高于未展示内容）。校准可以一定程度上缓解这种偏置，提升预测结果在未展示样本上的鲁棒性。

---

### **3. 校准的必要性分析**
校准是否必要，具体取决于你的应用场景：

| **场景**                                      | **校准是否必要**                    | **原因**                                                                 |
|-----------------------------------------------|-------------------------------------|--------------------------------------------------------------------------|
| **仅关心排序准确性**                          | 不必要                              | 排序仅需要分数的相对大小，校准后的绝对值对结果影响不大。                  |
| **需要概率解释（如CTR解释）**                 | 必要                                | 校准后的分数可以反映实际点击概率，增强模型的可解释性。                   |
| **多任务模型需要融合分数（如点击率+转化率）** | 必要                                | 统一分数尺度有助于加权融合不同目标。                                      |
| **需要跨模型融合（如召回模型与排序模型结合）** | 必要                                | 保证不同模型的输出分数在同一量纲内，有利于融合逻辑。                      |
| **广告竞价或收益优化**                        | 必要                                | 预测值需直接用于收益计算，因此校准后的概率分数是必要的。                   |

## 2. MMoE: Multi-gate Mixture-of-Experts
![figures/fig2.png](figures/fig9.jpg)

**MMoE**（**Multi-gate Mixture-of-Experts**）是一种多任务学习（Multi-task Learning）的深度学习模型架构，广泛应用于推荐系统、广告点击率预测等场景，尤其在多目标任务中非常高效。它通过共享专家网络（Experts）和任务特定的门控网络（Gate）来提升多任务建模能力，同时解决多任务之间的冲突问题。

---

### **1. MMoE 的核心思想**
在多任务学习中，任务之间可能存在竞争或冲突，例如推荐系统中“点击率预测”（CTR）和“转化率预测”（CVR）可能对特征的关注点不同。MMoE 引入了**专家共享机制**和**任务特定的门控网络**，使每个任务可以选择性地利用专家的知识，而不是完全共享或完全独立。

---

### **2. 模型结构**

MMoE 模型由以下几个模块组成：
**示例结构**：
```text
         Input Features
             ↓
      ┌───────────────┐
  Expert 1   Expert 2   ...  Expert N
      │         │               │
  ┌───▼───┐ ┌───▼───┐     ┌────▼────┐
  Gate 1   Gate 2   ...     Gate T
      │         │               │
 Task 1     Task 2           Task T
```

#### **（1）Shared Experts（共享专家网络）**
- 多个专家网络（通常是全连接神经网络）用于提取特征，捕捉特征的潜在表示。
- 专家网络是所有任务共享的，但并非每个任务都使用所有专家的输出。

#### **（2）Task-specific Gates（任务特定的门控网络）**
- 每个任务有一个独立的门控网络，用于对共享专家的输出进行加权。
- 门控网络输出一组权重，表示当前任务对各个专家的依赖程度。
- **Softmax** 用于生成归一化权重，最终对专家的输出进行加权求和，作为每个任务的输入。

#### **（3）Task Towers（任务特定网络）**
- 每个任务有自己的网络，用于根据门控网络输出的特征进一步优化。
- 任务特定网络的输出通常是该任务的预测值（例如点击率或转化率）。

---

### **3. 模型流程图**
MMoE 的整体架构可以总结如下：

```
输入特征
   ↓
共享专家网络（多个专家）
   ↓
任务 A 的门控网络  → 专家加权 → 任务 A 的特定网络 → 任务 A 的输出
任务 B 的门控网络  → 专家加权 → 任务 B 的特定网络 → 任务 B 的输出
...
```

- **共享专家**：负责提取基础特征表示。
- **任务门控**：为每个任务选择最合适的专家。
- **任务网络**：根据加权后的特征进行预测。

---

### **4. 代码实现（PyTorch）**

以下是一个简单的 MMoE 实现：

```python
import torch
import torch.nn as nn

class MMoE(nn.Module):
    def __init__(self, input_dim, expert_num, expert_hidden_dim, task_num, task_hidden_dim):
        super(MMoE, self).__init__()
        self.expert_num = expert_num
        self.task_num = task_num
        
        # 定义专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU()
            ) for _ in range(expert_num)
        ])
        
        # 定义任务门控网络
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, expert_num) for _ in range(task_num)
        ])
        
        # 定义任务特定网络
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_dim, task_hidden_dim),
                nn.ReLU(),
                nn.Linear(task_hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(task_num)
        ])
    
    def forward(self, x):
        # 专家网络的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, expert_num, hidden_dim]
        
        # 各任务的输出
        task_outputs = []
        for i, gate in enumerate(self.gates):
            gate_weights = torch.softmax(gate(x), dim=1)  # [batch_size, expert_num]
            
            # einsum:矩阵乘法
            # gate_weights: [batch_size, expert_num], 
            # expert_outputs: [batch_size, expert_num, hidden_dim]
            # gate_output: [batch_size, hidden_dim]
            gate_output = torch.einsum('be,beh->bh', gate_weights, expert_outputs)  # [batch_size, hidden_dim]
            task_output = self.towers[i](gate_output)  # [batch_size, 1]
            task_outputs.append(task_output)
        
        return task_outputs  # 每个任务的预测值

# 示例输入
input_dim = 64
expert_num = 4
expert_hidden_dim = 32
task_num = 3
task_hidden_dim = 16

model = MMoE(input_dim, expert_num, expert_hidden_dim, task_num, task_hidden_dim)
x = torch.rand(32, input_dim)  # 32个样本
outputs = model(x)

# 输出
for i, output in enumerate(outputs):
    print(f"Task {i+1} Output Shape: {output.shape}")
```

### 5. Polarize:极化现象
MMoE的Gate使用Softmax输出权重，而权重可能导致极化现象，即某个专家权重接近1，其他接近0。这样导致某些expert是死亡的，变成
普通多目标模型，而并没有融合多专家。

解决方案：专家Dropout
- 引入动态机制，在每次训练中随机屏蔽部分专家的输出（类似 Dropout），迫使门控网络选择更多的专家。这种方法可以有效减少专家极化现象。

## 3. PLE: Progressive Layered Extraction
**PLE（Progressive Layered Extraction）** 和 **MMoE（Multi-gate Mixture-of-Experts）** 是两种常见的多任务学习（MTL）框架，主要用于解决多任务推荐或多目标优化问题。这两种方法有一定相似性，但设计思想和应用场景有所不同。以下是两者的详细对比：

---

### 1. 基本概念
- **核心思想**：
  - 在 MMoE 的基础上，进一步将**共享专家**和**任务特定专家**分离，并引入分层特征提取结构。
  - 通过多层次专家网络对特征逐步提炼，解决任务冲突问题，同时保留任务间协同效应。
- **优点**：
  - 更好地缓解任务冲突问题。
  - 能同时建模任务相关性和任务独立性。
- **缺点**：
  - 模型复杂度更高，计算成本相较 MMoE 增加。

---

### **2. 网络架构**


---

#### **PLE 的架构**
1. **共享专家**提取可供所有任务共享的特征。
2. **任务特定专家**提取专属于每个任务的特定特征。
3. 每一层通过门控机制动态选择特征来源，并将上一层的输出逐层输入下一层，实现逐步特征提取。
4. 通过分层设计，逐步将特征分解为任务共享部分和任务特定部分。

**示例结构**：
```text
          Input Features
               ↓
 ┌──────── Shared Experts ────────┐
 │                                │
 Task A Experts          Task B Experts
       ↓                        ↓
      Gate A                  Gate B
       ↓                        ↓
   Task A Output            Task B Output
```

---




#### **PLE 示例代码**
```python
class PLE(nn.Module):
    def __init__(self, input_dim, expert_num, expert_dim, task_num):
        super(PLE, self).__init__()
        # 共享专家网络
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU()
            ) for _ in range(expert_num)
        ])
        # 每个任务的特定专家网络
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.ReLU()
                ) for _ in range(expert_num)
            ]) for _ in range(task_num)
        ])
        # 门控网络
        self.gates = nn.ModuleList([
            nn.Linear(input_dim, expert_num * 2) for _ in range(task_num)
        ])
        # 任务特定塔层
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, 1),
                nn.Sigmoid()
            ) for _ in range(task_num)
        ])

    def forward(self, x):
        # 获取共享专家的输出
        shared_outputs = torch.stack([expert(x) for expert in self.shared_experts], dim=1)
        task_outputs = []
        for i, task_expert in enumerate(self.task_experts):
            # 获取任务特定专家的输出
            task_specific_outputs = torch.stack([expert(x) for expert in task_expert], dim=1)
            # 拼接共享专家和任务特定专家的输出
            all_expert_outputs = torch.cat([shared_outputs, task_specific_outputs], dim=1)
            # 门控机制
            gate_weights = torch.softmax(self.gates[i](x), dim=1)
            gate_output = torch.einsum('be,bne->bn', gate_weights, all_expert_outputs)
            # 任务特定塔层输出
            task_outputs.append(self.towers[i](gate_output))
        return task_outputs
```

#### 关于Experts 和 Gates：
对于MMoE和PLE的Experts，一般来说每个下游任务有一个gate，该gate对所有experts进行加权。而PLE有一些shared experts用于所有任务，
还有一些task-specific experts用于特定任务。

## 4. 预估分数融合
### 1. 线性加权

简单加权：直接对不同任务的预估分数进行加权求和，得到最终的排序分数。
- score = p_click+w1*p_like+w2*p_collect+w3*p_share

点击率乘其他项的加权
- score = p_click*(1+w1*p_like+w2*p_collect+w3*p_share)

### 2. 快手的分数融合
![figures/fig2.png](figures/fig10.jpg)

### 3. 电商的分数融合
![figures/fig2.png](figures/fig11.jpg)

## 5. 视频播放建模
- 图文item排序主要依靠：点击，点赞，收藏，分享等行为，
- 视频播放建模主要依靠：播放时长，播放次数，播放完成率等指标。（直接做时长回归模型效果不好）
### 播放时长建模
![figures/fig2.png](figures/fig12.jpg)

- 上图所示，最右边的全连接层是播放时长的输出（其他是点击率，点赞率等）；
- 对z做sigmoid变化得到p，训练的时候用y=t/(1+t)作为label，用CE(y,p)作为损失函数;
- 推理的时候只使用exp(z)，因为CE会最小化p和y的差距，那么可以认为exp(z)和t的差距也会很小。即时长t=exp(z)。
- 通过这种方式，可以将时长回归问题转化为二分类问题，提高模型的泛化能力。
- 将exp(z)作为预估分数融合中的一项，影响视频item的排序。

### 完播率建模
- 回归方法：播放长度/视频长度作为label，p为预估播放率
  - loss = y*log(p)+(1-y)*log(1-p)
  - 如果p=0.73：预计播放长度为73%的视频长度

- 二分类：将完播80%的视频作为正样本，其他作为负样本，使用CE作为损失函数
  - 如果p=0.73：P(播放>=80%)=0.73，即73%的概率播放长度>=80%的视频长度

实际操作不可以直接用完播率作为融分公式的一项，因为完播率和播放时长有相关性，视频时长长的视频完播率可能较低。
- 对完播率预估值进行adjust：p_adjust = p/f(视频长度)，f(视频长度)是视频长度的函数，可以是线性函数，也可以是其他函数。
- 视频长度越长，f越小，p_adjust越大，即视频长度越长，完播率得分倾向于更高
- 将p_adjust作为融分公式的一项，影响视频item的排序。

## 6. 排序模型的特征
### 1. 用户画像 User Profile
- UID: 在召回，排序中进行embedding，一般用32维或64维向量
- 统计学属性：性别，年龄等
- 账号信息：新老用户，活跃度等
- 用户感兴趣类目：关键词，品牌等

### 2. 物品画像 Item Profile
- 物品id：embedding
- 发布时间：一般时间越长的物品，权重越低（更关注近期内容）
- GeoHash（经纬度编码），所在城市等
- 物品内容：标题，类目，关键词，品牌等（一般是分类变量，做embedding）
- 物品特征：字数，图片数，视频清晰度，标签是数等，反应item质量
- 内容信息量，图片美学等：使用cv，nlp模型对这些特征打分并融入item画像作为特征

### 3. 用户统计特征
- 用户在不同时间粒度下的点击数，点赞数等
- 用户分别对图文item，视频item的点击率等，反应对不同item的偏好
- 对item类目的点击率：比如对美妆类，科技类等细分领域维度

### 4. 笔记统计特征
- 笔记在不同时间的曝光数，点击数等
- 按照item受众分桶，比如item来自不同性别用户，不同年龄用户的点击
- 作者特征：作者发布的item数，粉丝数，消费指标（item的曝光数，点击数点赞数等）

### 5. 场景特征 Context
- 用户定位GeoHash
- 当前时刻：分段做embedding
- 是否是周末，节假日等
- 手机品牌，手机型号，操作系统（比如安卓和苹果用户的点赞率等指标有显著差异）

### 特征处理
- 离散特征：embedding
  - 用户id，itemId，作者ID
  - 类目，关键词，城市等
- 连续特征：分桶，变成离散特征
  - 年龄，笔记字数，视频长度等，先分桶为不同年龄组等离散特征，然后可以onehot或embedding
  - 点击数，曝光数等数值很大，可以做log变换：log(1+x)
    - 或者转换为点击率，点赞率，并做平滑处理

### 特征覆盖率
- 很多特征无法覆盖100%样本---存在缺失值；比如一些用户不填写年龄等
- 对于重要的特征，可以提高覆盖率来提高精排模型准度

![figures/fig2.png](figures/fig13.jpg)


## 7. 粗排 Pre-Ranking
前面部分的排序模型主要用于精排，需要高准度，单次推理代价大，同时样本量也比较小。
而粗排需要给更多item打分，单次推理代价要求小，可以牺牲一定的准度

一般前期融合（先对各种特征concatenation再输入一个shared bottom）用于精排阶段，而比如双塔模型等属于后期融合（分别将不同特征输入不同神经网络）用于粗排或召回

### 三塔模型
![figures/fig2.png](figures/fig14.png)

- 用户Tower：用户特征，场景特征
  - 对于一个用户只需要推理一次，所以用户塔可以很大，实际推理代价也不高
- 物品Tower：物品特征（静态）
  - 有n个item，每次给一个user做推荐则需要进行n次推理
  - 由于一般item属性较为稳定，可以缓存物品塔的输出向量
  - 一般线上不需要再推理，只有出现新item才需要推理
- 交叉Tower：统计特征，用户特征和物品特征交叉
  - 统计特征（点击率等）在每次交互后会变化，需要实时更新
  - 有n个item，必须做n次推理
  - 所以交叉塔需要足够小，推理代价低
- 三个Tower输出三个向量表征，对三个向量进行concatenation和cross
  - 把上述得到的一个向量分别输入不同的下游任务，输出点击率，点赞率，转发率等预估
  - 介于前期融合和后期融合之间
- 上层网络（用于输出点击率等指标的预估值）
  - 对每一个user的推荐请求，需要做n次推理，对n个item打分，代价大
  - 大部分粗排计算量都在上层网络部分

# Part 4: 特征交叉（Feature Cross）
## 1. Factorized Machine（FM）
FM是线性模型的改进，引入二阶交叉项来提升模型表达能力。FM模型的核心思想是将特征的交叉项分解为两个低维向量的内积，从而降低模型的复杂度。

## 2. DCN: Deep & Cross Network 深度交叉网络
DCN是一种结合了深度神经网络和特征交叉的模型，通过交叉层实现特征交叉，通过深度网络学习特征的高阶交叉。

之前提到的召回和排序模型中的`双塔模型`,`多目标学习模型`，`MMoE`等只是模型框架结构，其中的Tower和MMoE的shared bottom，Experts等都可以使用简单全连接网络或DCN或其他更复杂的网络结构。

![figures/fig2.png](figures/fig15.jpg)

### DCN的模型结构
DCN由两部分组成：
1. **交叉网络（Cross Network）**：
   - 显式学习特征的高阶交叉。
   - 通过逐层递归的方式，将输入特征进行多阶交叉组合。

2. **深度网络（Deep Network）**：
   - 使用多层感知机（MLP）学习特征的隐式交叉。
   - 包括多层全连接层，后接激活函数（如ReLU）。

3. **联合输出层**：
   - 将交叉网络和深度网络的输出进行融合（如拼接），最终接入一个全连接层，用于最终预测。


### 1. Cross Layer
深度交叉网络（Deep Cross Network, DCN）是一种用于特征交叉与组合的深度学习模型，常用于推荐系统和广告点击率预估任务中。DCN 的目标是通过高效地学习特征交叉来捕捉特征之间的非线性关系，从而提高模型的预测效果。
- x_i+1 = x_0 * (W*x_i + b_i) + x_i
- W*x_i + b_i是一个全连接层，x_0是输入特征，x_i是交叉层的输出，x_i+1是下一层的输入

![figures/fig2.png](figures/fig16.jpg)

### 2. Code Implementation
```python
class CrossLayer(nn.Module):
    """
    Cross Layer for explicit feature crossing
    """
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1))  # Learnable weight
        self.bias = nn.Parameter(torch.randn(input_dim))      # Learnable bias

    def forward(self, x0, xl):
        # Cross layer computation: x_{l+1} = x0 * (w^T * xl) + b + xl
        cross_term = torch.matmul(xl, self.weight)  # (batch_size, 1)
        cross_term = x0 * cross_term                # Element-wise product with x0
        return cross_term + self.bias + xl          # Add bias and residual connection

class CrossNetwork(nn.Module):
    """
    Cross Network: Stack of Cross Layers
    """
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        # 创建多个 CrossLayer，每一层的输入都是 (x0, xi)
        self.cross_layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x  # 保存初始输入 x0
        for layer in self.cross_layers:
            # 每一层接收 (x0, xi) 作为输入，x0 是初始输入，xi 是前一层的输出
            x = layer(x0, x)  # 每个 CrossLayer 的输入包括 x0 和当前的 x（上一层的输出）
        return x


class DeepNetwork(nn.Module):
    """
    Deep Network: Multi-Layer Perceptron (MLP) for implicit feature interaction
    """
    def __init__(self, input_dim, hidden_dims):
        super(DeepNetwork, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Optional: Dropout for regularization
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class DCNv2(nn.Module):
    """
    DCNv2: Combines Cross Network and Deep Network
    """
    def __init__(self, input_dim, cross_layers, hidden_dims):
        super(DCNv2, self).__init__()
        self.cross_network = CrossNetwork(input_dim, cross_layers)
        self.deep_network = DeepNetwork(input_dim, hidden_dims)
        self.output_layer = nn.Linear(input_dim + hidden_dims[-1], 1)  # Final output layer

    def forward(self, x):
        # Explicit feature crossing
        cross_out = self.cross_network(x)

        # Implicit feature interaction
        deep_out = self.deep_network(x)

        # Concatenate outputs from cross network and deep network
        combined = torch.cat([cross_out, deep_out], dim=1)

        # Output layer (e.g., for binary classification or regression)
        return torch.sigmoid(self.output_layer(combined))
```

## 3. LHUC(PPNet): Learning Hidden Unit Contributions
LHUC起源于语音识别（2016），后来常用于精排模型（PPNet）。LHUC的主要目标是通过为每个隐藏单元引入一个权重系数（即“贡献”），来提高模型对输入的适应性。这些贡献系数是可以训练的参数，模型通过学习这些系数，调整每个神经单元的权重，从而在不同的输入数据上表现出更好的灵活性和表达能力。

![figures/fig2.png](figures/fig17.jpg)

- item feature：用简单全连接层处理得到嵌入
- user feature：用MLP处理，并连接sigmoid，将结果*2（sigmoid结果在0-1，*2后结果在0-2）
- 两者输出的向量做element-wise相乘，得到最终的特征向量
- 可以继续堆叠上述结构，得到更复杂的模型
- 实际上是使用 用户特征 来作为 可学习的权重 来调整（或“缩放”） 物品（Item）特征，这一过程可以看作是 LHUC 的一种应用
- 模型可以根据不同用户的特征，动态调整每个物品的特征表示。这可以帮助模型根据每个用户的兴趣和行为，对物品的表示进行不同程度的调整
- 通过学习和优化这些权重，模型能够更好地适应不同用户的需求，尤其是在冷启动问题或数据稀疏的情况下，能够更灵活地调整每个用户和物品的特征交互。

## 4. SENet
**SENet**（**Squeeze-and-Excitation Networks**）是一种用于提升深度学习模型性能的网络架构，最初提出用于计算机视觉任务中的图像分类。SENet的核心思想是通过 **自适应地调整通道之间的权重**，来提升模型对重要特征的关注能力，从而增强模型的表达能力。它通过引入 **Squeeze-and-Excitation** 操作，在每个卷积层后加上一个 **自适应重标定机制**，使网络可以根据输入特征的不同，自动调整通道的重要性。

### **SENet的核心思想：**

1. **Squeeze（压缩）**：首先，通过全局平均池化将输入特征图的空间维度压缩为一个通道描述符。这个描述符反映了每个通道的重要性。
   
2. **Excitation（激励）**：接着，通过一个 **全连接层（FC layer）** 和一个 **sigmoid 激活函数**，生成每个通道的 **注意力系数**。这些系数会告诉模型每个通道的重要性，即该通道对最终输出的贡献。

3. **重标定**：最后，将 **激励系数** 与原始输入特征图的每个通道相乘，来重新调整各个通道的响应。这样，模型就能更关注重要的通道特征，并抑制不重要的通道特征。

### **SENet在推荐系统中的应用：**

虽然SENet最早是为计算机视觉任务设计的，但其自适应权重调整的思想同样可以应用到 **推荐系统** 中。推荐系统通常面临着多个特征之间的相互作用，而SENet的 **通道注意力机制** 可以帮助模型动态地选择和调整哪些特征对最终预测更为重要。

在推荐系统中，SENet的主要应用可能体现在以下几个方面：

1. **特征选择与加权**：
   - 在传统的推荐系统中，用户和物品的特征通常是通过嵌入层表示的，并通过神经网络进行处理。SENet可以通过 **自适应调整** 每个特征嵌入的权重，让模型更加关注与当前用户或物品相关的特征。

2. **增强特征交互**：
   - 推荐系统中，用户和物品特征的交互非常关键，尤其是对于深度神经网络模型。SENet通过自动选择重要的特征通道，可以增强模型对于特征交互的学习能力，从而更好地捕捉 **用户-物品的复杂关系**。

3. **提高个性化推荐的效果**：
   - 每个用户的兴趣和物品的特性不同，SENet的注意力机制可以让模型更加关注用户当前偏好的物品特征和行为模式，提升 **个性化推荐** 的效果。

# Part 5: 用户行为建模
## 1. LastN模型
- 主要使用用户最后交互过的N个item进行embedding（包括item的id，以及其他物品特征等）
- 得到你N个嵌入向量，对N个向量取平均得到一个向量，表示用户最近感兴趣的物品
- 将LastN特最终的特征与其他特征cat，输入到召回，排序模型中
- LastN可用于召回双塔，粗排三塔，精排等模型
- LastN包括点击，点赞，收藏等行为的item。

## 2. DIN: Deep Interest Network
用加权平均代替LastN的简单平均，类似注意力机制。
- 对于某候选物品（比如粗排的结果item是精排的候选item），计算与用户LastN物品的相似度
- 用相似度作为权重，计算LastN物品的加权平均，得到一个向量
- 把上面的结果想了作为一种用户特征输入排序模型

### 1. **DIN 模型的核心思想**

DIN 模型的核心是**动态兴趣建模**，即模型能够根据当前推荐的物品（或广告），动态地从用户的历史行为中挑选出与当前物品最相关的历史行为，并对这些历史行为给予更多关注，从而更准确地建模用户的兴趣。其基本思想是：

- **兴趣历史选择**：模型根据当前的推荐物品，选择用户历史行为中与当前物品相关的部分作为动态兴趣。
- **用户行为建模**：通过注意力机制或者其他方法，将用户与物品之间的交互历史进行建模，以便更好地捕捉用户兴趣的时序性变化。
  

### 2. **DIN 模型的注意力机制**

在 DIN 模型中，**注意力机制**（Attention Mechanism）是核心组成部分，它用于对用户历史行为进行加权选择。具体而言，DIN 通过以下方式进行兴趣历史的加权：

- **查询（Query）**：候选物品的嵌入表示。
- **键（Key）**：用户历史行为中的物品嵌入表示。
- **值（Value）**：用户历史行为中每个物品的嵌入表示。

注意力机制通过计算当前物品与历史物品的相似度（通常使用点积）来决定历史行为的加权系数。然后，基于这些加权系数，将用户历史行为向量加权求和，得到最终的动态兴趣表示。

### 3. **DIN 模型的具体结构图**

以下是 DIN 模型的一般结构：

1. **输入：**
   - 用户的历史行为序列：包括历史物品的 ID。
   - 当前推荐的物品的 ID。

2. **嵌入层：**
   - 用户和物品的 ID 都经过嵌入层（Embedding）转化为低维向量。
   
3. **动态兴趣建模：**
   - 当前推荐的物品作为查询（Query），与用户历史行为中的每个物品进行相似度计算，得到注意力权重。
   - 然后，使用这些权重对用户历史行为进行加权求和，得到动态兴趣向量。

4. **兴趣匹配：**
   - 将动态兴趣向量与当前推荐的物品的嵌入向量进行匹配（通常是通过点积或其他相似度计算方法）。

5. **输出：**
   - 将匹配结果输入到一个全连接层，输出用户对该物品的兴趣评分或点击概率。

**DIN 模型的代码实现概述**

```python
class DIN(nn.Module):
    def __init__(self, user_size, item_size, embedding_dim, hidden_dim):
        super(DIN, self).__init__()
        
        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(user_size, embedding_dim)
        self.item_embedding = nn.Embedding(item_size, embedding_dim)
        
        # 历史行为的嵌入层
        self.history_item_embedding = nn.Embedding(item_size, embedding_dim)
        
        # 动态兴趣建模中的注意力机制
        self.attention_weight = nn.Linear(embedding_dim, 1)
        
        # 全连接层，用于预测输出
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, user_id, item_id, history_item_ids):
        # 获取用户和当前物品的嵌入向量
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        
        # 获取用户历史物品的嵌入向量
        history_emb = self.history_item_embedding(history_item_ids)  # (batch_size, history_len, embedding_dim)
        
        # 计算注意力得分
        attention_score = torch.matmul(history_emb, item_emb.unsqueeze(2))  # (batch_size, history_len, 1)
        attention_score = torch.squeeze(attention_score, dim=2)  # (batch_size, history_len)
        attention_weights = F.softmax(attention_score, dim=1)  # (batch_size, history_len)
        
        # 使用注意力权重对历史物品嵌入加权
        # 沿着dim=1：history_len的方向，对每个历史物品嵌入进行加权求和
        weighted_history_emb = torch.sum(history_emb * attention_weights.unsqueeze(2), dim=1) # (batch_size, embedding_dim)
        
        # 将当前物品的嵌入和加权历史嵌入连接
        interaction = torch.cat([item_emb, weighted_history_emb], dim=1) # (batch_size, embedding_dim * 2)
        
        # 全连接层预测评分
        output = self.fc(interaction) # (batch_size, 1)
        return output
```

