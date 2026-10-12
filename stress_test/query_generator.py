"""
Agent_online 压力测试 - 动态查询生成器
=============================================
问题背景：
    如果压测只用固定的几十个问题反复循环，服务预热几秒后几乎全部命中
    Redis 缓存（实测流式接口缓存命中率高达 98%），根本压不到真正的
    RAG 检索 + LLM 生成回源路径，压测结果失真。

真实线上流量特征：
    - 少量"热点问题"反复出现  → 命中缓存（约 30~45%）
    - 大量"长尾问题"各不相同  → 缓存未命中，走完整 RAG+LLM（约 55~70%）

本模块通过【模板 + 多维槽位组合】动态生成海量近乎不重复的医学问题，
真实模拟长尾流量，把缓存命中率拉回线上真实水平，让压测测出真实性能。

组合空间估算：
    症状/疾病(≈120) × 提问模板(≈25) × 修饰维度(人群/病程/部位/程度 各≈8)
    → 理论可生成数十万种不同问题，足以覆盖任意压测时长而不重复。
"""

import hashlib
import random
import threading

# ==================== 槽位词库（真实医学语料）====================

# —— 疾病 / 症状主体（按科室覆盖，模拟真实咨询分布）——
DISEASES_SYMPTOMS = [
    # 心血管
    "高血压", "低血压", "冠心病", "心律不齐", "心肌梗死", "心绞痛", "动脉硬化",
    "高血脂", "心力衰竭", "房颤", "心肌炎",
    # 内分泌 / 代谢
    "糖尿病", "甲亢", "甲减", "甲状腺结节", "痛风", "高尿酸", "肥胖症", "低血糖",
    "骨质疏松", "更年期综合症",
    # 消化系统
    "胃炎", "胃溃疡", "胃食管反流", "肠易激综合征", "便秘", "腹泻", "脂肪肝",
    "胆结石", "肝功能异常", "幽门螺杆菌感染", "痔疮", "阑尾炎",
    # 呼吸系统
    "感冒", "流感", "支气管炎", "肺炎", "哮喘", "慢阻肺", "肺结核", "咽炎",
    "扁桃体炎", "鼻窦炎", "过敏性鼻炎",
    # 神经 / 精神
    "偏头痛", "失眠", "抑郁症", "焦虑症", "神经衰弱", "面瘫", "帕金森", "癫痫",
    "眩晕症", "三叉神经痛",
    # 骨科 / 疼痛
    "颈椎病", "腰椎间盘突出", "肩周炎", "关节炎", "腰肌劳损", "骨折", "半月板损伤",
    "坐骨神经痛", "足底筋膜炎", "网球肘",
    # 泌尿 / 肾
    "肾结石", "尿路感染", "前列腺炎", "前列腺增生", "肾炎", "尿失禁", "膀胱炎",
    # 皮肤
    "湿疹", "荨麻疹", "银屑病", "痤疮", "带状疱疹", "灰指甲", "脚气", "白癜风",
    "脱发", "皮肤过敏",
    # 妇科 / 儿科
    "月经不调", "痛经", "多囊卵巢综合征", "乳腺增生", "妊娠糖尿病", "小儿腹泻",
    "小儿发烧", "手足口病", "湿疹",
    # 眼耳口
    "结膜炎", "白内障", "青光眼", "干眼症", "中耳炎", "耳鸣", "口腔溃疡", "牙周炎",
    # 肿瘤 / 血液
    "贫血", "白血病", "淋巴瘤", "肺癌", "胃癌", "乳腺癌", "甲状腺癌",
]

# —— 具体症状描述（用于"症状类"提问）——
SYMPTOM_PHRASES = [
    "头晕", "头痛", "恶心", "呕吐", "胸闷", "心悸", "气短", "乏力", "发烧", "咳嗽",
    "咳痰", "腹痛", "腹胀", "反酸", "食欲不振", "失眠多梦", "手脚发麻", "关节疼痛",
    "腰酸背痛", "视力模糊", "耳鸣", "皮肤瘙痒", "尿频尿急", "便血", "体重下降",
    "盗汗", "水肿", "口干口苦", "怕冷", "手抖",
]

# —— 人群修饰（尽量不留空，以扩大组合空间、降低长尾重复率）——
POPULATIONS = [
    "老年人", "儿童", "孕妇", "婴幼儿", "青少年", "上班族", "糖尿病患者",
    "高血压患者", "术后患者", "哺乳期妇女", "中年男性", "更年期女性",
    "退休老人", "长期熬夜的人", "体质虚弱者", "肥胖人群", "白领", "学生",
    "备孕女性", "老烟民", "久坐族", "重体力劳动者",
]

# —— 病程 / 时间修饰 ——
DURATIONS = [
    "刚出现", "持续一周", "反复发作", "长期", "夜间加重", "早上起床时",
    "运动后", "饭后", "劳累后", "季节交替时", "最近半个月", "断断续续",
    "情绪激动后", "受凉后", "熬夜后", "吃辛辣食物后", "阴雨天",
    "每到下午", "半夜", "空腹时",
]

# —— 身体部位修饰 ——
BODY_PARTS = [
    "左侧", "右侧", "上腹部", "下腹部", "后背", "膝盖", "脖子", "太阳穴",
    "关节", "小腿", "胸口", "肩膀", "腰部", "脚踝", "手腕", "头顶",
    "两侧", "整个", "局部", "下肢",
]

# —— 程度 / 性质修饰 ——
DEGREES = [
    "轻微", "严重", "间歇性", "持续性", "阵发性", "钝痛", "刺痛", "隐隐", "剧烈",
    "偶尔", "频繁", "轻度", "中度", "明显", "反复", "突然", "逐渐加重",
    "时轻时重", "说不清的", "难以忍受的",
]

# ==================== 提问模板 ====================
# {d}=疾病/症状主体  {s}=症状短语  {p}=人群  {t}=病程  {b}=部位  {g}=程度

# —— 疾病主体类模板（占长尾主流）——
# 每个模板都至少嵌入一个高基数修饰槽位（{p}人群 / {t}病程 / {g}程度），
# 让组合空间成倍放大，长尾问题近乎不重复 → 逼近线上真实的低缓存命中率。
DISEASE_TEMPLATES = [
    "{p}的{d}应该怎么治疗？",
    "{p}{d}的早期症状有哪些？",
    "{p}得了{d}要注意什么？",
    "{t}的{d}是什么原因引起的？",
    "{p}患{d}平时饮食上要注意哪些？",
    "{p}的{d}能不能根治？",
    "{p}{d}需要长期吃药吗？",
    "{t}的{d}和哪些疾病容易混淆？",
    "{t}的{d}严重吗？",
    "{p}的{d}会遗传吗？",
    "{p}{d}可以运动吗？",
    "{p}{d}用什么药比较好？",
    "{p}的{d}做哪些检查可以确诊？",
    "{t}的{d}会有哪些并发症？",
    "{p}{d}恢复期需要多久？",
    "{p}的{d}挂什么科？",
    "{t}的{d}可以用中医调理吗？",
    "{p}的{d}日常怎么预防？",
    "{t}的{d}该怎么缓解？",
    "{p}得了{d}生活方式要怎么调整？",
    "{p}{t}的{d}要不要住院？",
    "{p}的{d}多久能好？",
    "{t}的{d}挂号看什么门诊？",
]

# —— 症状类模板（模拟"我有xx症状"的真实咨询）——
# 症状模板天然消耗多个高基数槽位（{p}{b}{g}{t}{s}），组合空间极大。
SYMPTOM_TEMPLATES = [
    "{p}{b}{g}{s}是怎么回事？",
    "{p}{t}{s}需要看医生吗？",
    "{g}{s}可能是什么病？",
    "{p}最近总是{b}{s}怎么办？",
    "{b}{g}{s}是什么原因？",
    "{p}{t}出现{s}要紧吗？",
    "{p}一{t}就{s}是什么问题？",
    "{p}{s}还伴随{s2}是什么情况？",
    "{b}{s}还{s2}该挂什么科？",
    "{p}{b}{t}{s}严重吗？",
    "{g}{s}并且{s2}怎么办？",
    "{p}{t}{b}{s}要做什么检查？",
]

# ==================== 热点问题（模拟高频重复，命中缓存）====================
# 这部分问题固定，被反复请求，模拟线上真实的"高频热点"（如常见病咨询）
# 数量刻意保持较少，命中缓存符合真实场景（热点约占总流量 25%）
HOT_QUESTIONS = [
    "高血压怎么治疗？",
    "糖尿病的早期症状有哪些？",
    "感冒了吃什么药好？",
    "失眠怎么办？",
    "痛风发作时如何缓解？",
    "颈椎病怎么锻炼？",
    "胃炎的饮食注意事项？",
    "高血脂如何通过饮食调理？",
    "小孩发烧怎么物理降温？",
    "头疼是什么原因引起的？",
]

# ==================== 中医古籍问题（走 ancient_books 集合）====================
TCM_QUESTIONS = [
    "黄帝内经对养生有什么论述？",
    "伤寒论中的经方有哪些？",
    "本草纲目记载的常用中药？",
    "针灸的主要穴位有哪些？",
    "中医的四诊法是什么？",
    "五脏六腑的对应关系是什么？",
    "中医如何看待湿气重？",
    "阴虚和阳虚有什么区别？",
    "中医怎么调理脾胃虚弱？",
    "经络学说讲的是什么？",
]

# ==================== 非医学 / 边界问题（测试路由分类）====================
NON_MEDICAL_QUESTIONS = [
    "你好", "在吗", "你是谁？", "你叫什么名字",
    "讲个笑话", "今天天气怎么样？", "1+1等于几？",
    "谢谢你", "再见", "你能做什么？",
]


# ==================== 生成器 ====================

class QueryGenerator:
    """
    动态查询生成器（线程安全）。

    通过流量模型混合以下几类查询，模拟真实线上流量分布：
      - 长尾疾病问题（模板+槽位动态生成，几乎不重复）→ 缓存未命中
      - 长尾症状问题（同上）                        → 缓存未命中
      - 热点问题（固定小集合，反复出现）             → 缓存命中
      - 中医古籍问题                                 → 部分命中
      - 非医学问题                                   → 静态路由
    """

    def __init__(self, traffic_model=None, seed=None):
        """
        Args:
            traffic_model: dict，各类查询的流量占比（和应≈1.0）。
                           不传则使用默认真实流量模型。
            seed: 随机种子（用于可复现测试）；None 表示每次随机。
        """
        # 默认流量模型：长尾占大头，热点约 25%，符合线上真实分布
        self.traffic_model = traffic_model or {
            "longtail_disease": 0.45,   # 长尾疾病问题（回源）
            "longtail_symptom": 0.20,   # 长尾症状问题（回源）
            "hot": 0.25,                # 热点问题（命中缓存）
            "tcm": 0.05,                # 中医古籍
            "non_medical": 0.05,        # 非医学/边界
        }
        self._validate_model()

        # 每个线程独立的 Random 实例，避免多线程竞争同一全局随机状态
        self._local = threading.local()
        self._seed = seed

        # 去重统计（可选，用于观测生成的唯一性）
        self._generated_count = 0
        self._unique_hashes = set()
        self._stats_lock = threading.Lock()

    def _validate_model(self):
        """校验流量模型占比之和接近 1.0"""
        total = sum(self.traffic_model.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"流量模型占比之和应为 1.0，当前为 {total:.3f}: {self.traffic_model}"
            )

    def _rng(self) -> random.Random:
        """获取当前线程专属的随机数生成器"""
        if not hasattr(self._local, "rng"):
            # 每个线程用不同种子（基于线程标识），保证线程间不同步
            if self._seed is not None:
                thread_seed = self._seed + threading.get_ident()
            else:
                thread_seed = None
            self._local.rng = random.Random(thread_seed)
        return self._local.rng

    def _fill_template(self, template: str, rng: random.Random) -> str:
        """
        用随机槽位填充模板，并清理空槽位残留的病句。

        例如模板 "{t}的{d}该怎么缓解？" 当 {t} 为空时会得到
        "的神经衰弱该怎么缓解？"（以"的"开头的病句），需要清理。
        """
        text = template.format(
            d=rng.choice(DISEASES_SYMPTOMS),
            s=rng.choice(SYMPTOM_PHRASES),
            s2=rng.choice(SYMPTOM_PHRASES),
            p=rng.choice(POPULATIONS),
            t=rng.choice(DURATIONS),
            b=rng.choice(BODY_PARTS),
            g=rng.choice(DEGREES),
        )
        return self._cleanup(text)

    @staticmethod
    def _cleanup(text: str) -> str:
        """清理空槽位导致的病句：多余空格、句首孤立助词。"""
        text = text.replace("  ", " ").strip()
        # 去除因空槽位残留在句首的助词（如 "的神经衰弱..." → "神经衰弱..."）
        while text and text[0] in "的了在时后":
            text = text[1:]
        return text.strip()

    def _gen_longtail_disease(self, rng: random.Random) -> str:
        """生成长尾疾病问题"""
        template = rng.choice(DISEASE_TEMPLATES)
        return self._fill_template(template, rng)

    def _gen_longtail_symptom(self, rng: random.Random) -> str:
        """生成长尾症状问题"""
        template = rng.choice(SYMPTOM_TEMPLATES)
        return self._fill_template(template, rng)

    def next_query(self) -> tuple:
        """
        按流量模型生成下一个查询。

        Returns:
            (question: str, category: str)
            category ∈ {longtail_disease, longtail_symptom, hot, tcm, non_medical}
        """
        rng = self._rng()
        r = rng.random()

        # 按流量模型累积概率选择类别
        cumulative = 0.0
        category = "longtail_disease"  # 兜底
        for cat, ratio in self.traffic_model.items():
            cumulative += ratio
            if r <= cumulative:
                category = cat
                break

        # 按类别生成问题
        if category == "longtail_disease":
            question = self._gen_longtail_disease(rng)
        elif category == "longtail_symptom":
            question = self._gen_longtail_symptom(rng)
        elif category == "hot":
            question = rng.choice(HOT_QUESTIONS)
        elif category == "tcm":
            question = rng.choice(TCM_QUESTIONS)
        elif category == "non_medical":
            question = rng.choice(NON_MEDICAL_QUESTIONS)
        else:
            question = self._gen_longtail_disease(rng)

        # 记录去重统计
        self._track(question)
        return question, category

    def next_question(self) -> str:
        """便捷方法：只返回问题文本（不含类别）"""
        return self.next_query()[0]

    def hot_question(self) -> str:
        """获取一个热点问题（用于专门的缓存命中验证场景）"""
        return self._rng().choice(HOT_QUESTIONS)

    def _track(self, question: str):
        """记录生成统计（用于观测唯一率）"""
        h = hashlib.md5(question.encode("utf-8")).hexdigest()
        with self._stats_lock:
            self._generated_count += 1
            self._unique_hashes.add(h)

    def stats(self) -> dict:
        """
        返回生成统计。
        unique_ratio 越接近 1.0 说明长尾越丰富、重复越少。
        """
        with self._stats_lock:
            total = self._generated_count
            unique = len(self._unique_hashes)
        return {
            "total_generated": total,
            "unique_questions": unique,
            "unique_ratio": round(unique / total, 4) if total > 0 else 0.0,
        }


# ==================== 模块级默认单例 ====================
# 各 locustfile 直接 import 使用，保证全局共享统计
default_generator = QueryGenerator()


# ==================== 自测 / 演示 ====================
if __name__ == "__main__":
    print("=" * 64)
    print("  动态查询生成器 - 自测演示")
    print("=" * 64)

    gen = QueryGenerator(seed=42)

    # 演示生成 20 个样例
    print("\n  [样例] 生成 20 个查询：")
    for i in range(20):
        q, cat = gen.next_query()
        print(f"    {i+1:>2}. [{cat:<18}] {q}")

    # 唯一率测试：生成 5000 个，观察重复情况
    print("\n  [唯一率测试] 生成 5000 个查询...")
    gen2 = QueryGenerator()
    category_count = {}
    for _ in range(5000):
        _, cat = gen2.next_query()
        category_count[cat] = category_count.get(cat, 0) + 1

    stats = gen2.stats()
    print(f"    总生成: {stats['total_generated']}")
    print(f"    唯一问题: {stats['unique_questions']}")
    print(f"    唯一率: {stats['unique_ratio']:.1%}")
    print(f"\n  [流量分布]")
    for cat, count in sorted(category_count.items(), key=lambda x: -x[1]):
        print(f"    {cat:<20}: {count:>5} ({count/5000:.1%})")

    # 预估缓存命中率：只有 hot 类 + 少量重复的长尾会命中
    hot_ratio = category_count.get("hot", 0) / 5000
    print(f"\n  [预估] 热点问题占比 ≈ {hot_ratio:.1%}")
    print(f"         → 缓存命中率预计落在 30~45% 区间（贴近线上真实值）")
    print("=" * 64)
