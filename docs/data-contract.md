# 业务资料约定

日期为公历日；period为参考期，released为首次可得日期，revision在同指标参考期内递增。flow按期汇总，stock表示期末存量；不得把不同单位直接相加。数据为发布版本资料，缺行表示没有该次发布。

每行id在该表内唯一，文件顺序不代表业务事件顺序。清单schema=1，各表登记文件名、字节长度、SHA256、行数及完整字段集合，未知版本或内容变化应重新登记。金额及测量数值均为独立资料，不代表任何现实机构的经营数据。

- series：id、frequency、aggregation、unit。
- releases：id、series、period、released、revision、value。
- decisions：id、day、target。

normal与boundary是相互独立的资料包，不能直接拼接重复id。引用字段按本项目domain.py的REFERENCES登记。读取检查只检查完整性、身份及这些直接引用，不替代领域分析中的可识别性、时点可见性或可行性判断。
