# Author Clarification Request — PA-001 and PA-004

`AUTHOR_REPLY_STATUS=NOT_CONTACTED`  
`PRIMARY_CONTACT=Simin Peng, corresponding author`  
`CONTACT_EVIDENCE=Target paper PDF p.1, asterisk footnote and “E-mail address: siminpeng@ycit.edu.cn (S. Peng).”`  
`SENDING_STATUS=HUMAN_REVIEW_REQUIRED__NOT_SENT_BY_CODEX`

## ENGLISH_VERSION

**Subject:** Clarification request for faithful reproduction of Energy 342 (2026) 139463

Dear Prof. Peng,

We are preparing a faithful academic reproduction of your paper, “Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach” (*Energy* 342 (2026) 139463; https://doi.org/10.1016/j.energy.2025.139463). Two protocol details would help us reproduce the experiments without introducing assumptions.

**Question 1 (SOH denominator).** The paper describes SOH as residual capacity relative to nominal capacity. For each experimental dataset, what exact capacity value was used as the denominator for the SOH label: the nominal rated capacity, the first/initial measured capacity, or another specified reference capacity?

**Question 2 (target-data protocol).** How should the chronological 60%/20%/20% target-battery split in Section 4 be related to the first 20% of total target-battery life used as the original small-sample training set in Section 4.2? Was the latter nested within the 60% training partition, a separate experimental setting, or another arrangement? If possible, could you indicate which target-cycle ranges were used for temporal-model training, validation, testing, and the SCF-GAN-augmented samples?

These details will be used only for faithful academic reproduction. Thank you very much for your time.

Sincerely,  
[Name / affiliation]

**Word count (email body excluding subject/signature): 183**

## CHINESE_REFERENCE_VERSION

**主题：** 请求澄清 Energy 342 (2026) 139463 的严格复现实验协议

彭教授您好：

我们正在对您的论文《Data-augmented SOH estimation for lithium-ion batteries under small-sample conditions: A hybrid STL-transformer-TimesNet approach》（*Energy* 342 (2026) 139463；DOI: 10.1016/j.energy.2025.139463）进行严格的学术复现。为避免自行引入假设，想请您澄清两个实验协议细节。

**问题 1（SOH 分母）：** 论文将 SOH 描述为相对于额定容量的剩余容量。各实验数据集构造 SOH 标签时，具体使用了哪个容量值作为分母：额定容量、首次/初始实测容量，还是其他指定参考容量？

**问题 2（目标电池数据协议）：** 第 4 节目标电池按时间顺序划分的 60%/20%/20%，与第 4.2 节“目标电池总寿命前 20% 循环构成原始小样本训练集”之间，具体是什么关系？后者是否嵌套于 60% 训练分区中、是独立实验设置，还是另有安排？如方便，也请说明 temporal model 的训练、验证、测试，以及 SCF-GAN 增强样本分别使用了目标电池的哪些循环区间。

这些信息仅用于忠实的学术复现。非常感谢您的时间。

此致  
敬礼  
[姓名 / 单位]

中文版仅供 Human 核对语义；建议发送英文版。
