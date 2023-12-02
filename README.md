
### 自我介绍
Hi，我是项健（英文名：Charles），自然语言处理算法工程师。 

### 工作经历 1

2023年4月-至今，自然语言处理算法工程师

**项目描述：** 面向券商/上市公司客户，在金融场景下，通过对金融文档材料聚类和抽取帮助客户进行自动化底稿管理，审计报告合规，信披公告检索等。个人负责多模态NLP金融文档聚类算法设计和优化。
1. 采用pipeline管道式算法流程，将聚类算法分为OCR和NLP两个模块。基于分层聚类算法总体框架，利用凝聚式和分裂式聚类两种方法，经过基于Faiss向量检索库k-近邻检索聚类，LCS最长公共子序列聚类，文档视觉特征聚类，再召回，标题分割聚类等7个步骤进行聚类。
2. 基于对比学习利用Sentence-BERT模型，通过构建不同金融文档版式类别的正负样本对微调模型，用来生成具有语义信息的句向量，进而利用欧式距离进行文档相似度计算。多项目准确率平均提升2.5%，召回率提升1.5%，F1提升2.24%。
3. 分析海量金融文档数据特点，总结规律特征，进行特征工程特征提取，分别优化LCS最长公共子序列聚类策略，标题分割聚类策略，再召回策略来解决小样本簇未召回问题和不同金融文档版式混淆问题，多项目F1平均从91%提升至97%。

### 工作经历 2

2021年3月 - 2023年3月，技术经理，北京大学

**项目描述：** 负责利用深度学习、自然语言处理技术做开源许可证文本自动化合规冲突检测算法设计，文本大数据分析，代码大数据研发 工作。
1. 基于PyTorch框架细粒度解析许可证文档做小样本命名实体识别和关系抽取任务，结合对比学习（Contrastive Learning)、提示学 习（Prompt Learning）等方法在小样本场景下识别开源许可证权利、义务、行为、条件等实体及实体间关系，F1分数超过85%。 2. 在T5-base预训练模型基础上，基于prompt 的通用联合信息抽取UIE框架，构建结构提取语言SEL，采用span抽取方式抽取命名实 体和实体关系开始和结束位置。
3. 对TB级文本数据做语义匹配，运用规则和模版方法自动对齐用户和SPDX官方许可证，解决70%语义一致性问题。
4. 基于数据挖掘技术设计多源漏洞挖掘、漏洞智能修复算法，通过定时任务自动更新漏洞库，减少30%运维工作量。 


### 工作经历 3
2020年3月-2021年2月， 推荐算法工程师，纽约

**项目描述：** 该项目是某电商网站畅销榜单、详情页相似推荐、购物车页面搭配推荐推荐算法迭代。目的是通过个性化推荐来提高网站商
品销售的有效转化率，改进用户体验。
1. 参与电商平台个性化推荐系统召回、排序算法设计，探索用户推荐功能场景。
2. 多路召回和排序算法设计，包括基于规则的标签召回、热门商品召回；协同过滤CF召回，U2U、 i2i；DSSM双塔语义向量召回等
模型。利用GBDT和LR排序算法部署精排模型，用户商品点击率提高10%。
3. 从消息队列中获取用户行为数据，利用K-means聚类算法进行用户群组聚类划分，生成用户画像。
4. 结合用户画像和标签-商品倒排索引（Elasticsearch）, 设计基于用户画像的推荐算法，给用户推荐TOP N商品。
   
**我的技术栈如下**：
**个人优势：** 有金融场景下多模态文档聚类算法、自然语言处理信息抽取算法和电商推荐算法经验。
**编程语言**：JAVA, Python

**框架**: PyTorch, Spring Boot, Spark, 消息队列(Kafka).  

**算法**： 深度学习/自然语言处理：迁移学习Transformer模型、Bert预训练语言模型、Sentence-BERT, 序列标注命名实体识别，对比学习（Contrastive Learning），提示学习(Prompt Learning); 数据挖掘算法: 分层聚类, K-means聚类，文本相似度算法；常见推荐算法/召回算法：协同过滤+DSSM双塔, Faiss向量检索；排序算法：GBDT+LR

**数据库**：关系型数据库SQL(MySQL), 非关系型数据库NoSQL(MongoDB), Spring Data JPA, Spring Data MongoDB

**开发工具和运维**: Maven, Gradle, Eclipse, JUnit, Git, Docker.

**如果你对我有兴趣，欢迎联系。 ** 

**邮箱地址**：<3426522815@qq.com>

**领英地址**：<https://www.linkedin.com/in/jian-xiang-profile/>

### Intro
Hi，My name is Jian Xiang(English name: Charles), NLP Engineer in Beijing, China.

### Work Experience 1

** Project description：**For securities firms/listed company customers, in the financial scenario, through clustering and extraction of financial document materials, it helps customers perform automated draft management, audit report compliance, information disclosure and announcement retrieval, etc. Personally responsible for the design and optimization of multi-modal OCR+NLP financial document clustering algorithms.

1. Using the pipeline pipeline algorithm process, the clustering algorithm is divided into two modules: OCR and NLP. Based on the overall framework of the hierarchical clustering algorithm, using two methods of agglomerative and split clustering, after k-nearest neighbor retrieval clustering based on the Faiss vector retrieval library, LCS longest common subsequence clustering, document visual feature clustering, and then Recall, title segmentation clustering and other 7 steps for clustering.
   
2. Using the Sentence-BERT model based on comparative learning, fine-tune the model by constructing positive and negative sample pairs of different financial document layout categories to generate sentence vectors with semantic information, and then use Euclidean distance to calculate document similarity. The average multi-item accuracy rate increased by 2.5%, the recall rate increased by 1.5%, and the F1 increased by 2.24%.
   
3. Analyze the characteristics of massive financial document data, summarize the regular characteristics, perform feature engineering feature extraction, respectively optimize the LCS longest common subsequence clustering strategy, title segmentation clustering strategy, and recall strategy to solve the problem of non-recall of small sample clusters and different For financial document layout confusion, the average multi-project F1 increased from 91% to 97%

### Work Experience 2

**Project description:** Responsible for utilizing Deep Learning and Natural Language Processing technology to do open source license automated compliance system, text mining to help customers avoid software knowledge right litigation.

1.Parsed open source license docment to do Named Entity Recognition(NER) and Relation Extraction in Few-Shot scenario based on PyTorch framework. Combined with Contrastive Learning and Prompt Learning to recognize open source license rights, obligations, behaviors and conditions entities and relationships between entities. F1 score is above 85%. 
2. On the basis of T5-base pretrained model, exploited Universal Information Extraction(UIE) framework with Prompt Learning to construct Structured Extraction Language in order to spot locations of named entities and their relations ground on Span Extraction. 
3. Utilized text analysis modules to align users' license with SPDX official license to solve 70% semantic consistency problems.

### Work Experience 3
**Project description:** This project utilized iterative recommendation algorithms for an e-commerce website's best-selling list, similar recommendation on detail pages, and matching recommendation on shopping cart pages. The purpose is to improve the effective conversion rate of product sales and improve user experience through personalized recommendations.

1. Participated in the recall and ranking algorithm design of the personalized recommendation system in the e-commerce platform, and explore user recommendation function scenarios.
   
2. Multi-channel recall and sorting algorithm design, including rule-based label recall, popular product recall; collaborative filtering CF recall, U2U, i2i; DSSM twin-tower semantic vector recall Model. Using the GBDT and LR sorting algorithms to deploy the fine-sorting model, the click-through rate of users' products increased by 10%.
   
3. Obtained user behavior data from the message queue, use K-means clustering algorithm to cluster and divide user groups, and generate user portraits.
  
4. Combined user portraits and tag-product inverted index (Elasticsearch), designed a recommendation algorithm based on user portraits to recommend TOP N products to users.



**If you have interests on me, welcome to contact me through the email below.** 

**Email Address**：<3426522815@qq.com>

**linkedin**：<https://www.linkedin.com/in/jian-xiang-profile/>



### My Tech Stack is as follows:
**Languages**: JAVA, Python, Linux

**Framework**: PyTorch, Spring Boot, Spark, Message Queue(Kafka).

**Algorithm**: Deep Learning/Natural Language Processing: Transfer Learning (Transformer Model)、Bert (Pre-trained language model)、Sentence-BERT, Information Extraction(Named entity Recognition)，Contrastive Learning，Prompt Learning; Data Mining: Hierarchical clustering, K-means clustering，Text similarity algorithm; Recommendation algorithm/recall algorithm: collaborative filtering, DSSM, Faiss similarity search；Ranking Algorithm:GBDT+LR

**Database**:  SQL(MySQL), NoSQL(MongoDB), Spring Data JPA, Spring Data MongoDB

**Development Tool & DevOps**: Maven, Gradle, Eclipse, JUnit, Git, Docker.



