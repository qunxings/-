
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# ------------------------------------------------
# 0. 统一路径（只改这里）
# ------------------------------------------------
ROOT_DIR        = r"D:\study\作业\文本分类"
DATA_DIR        = os.path.join(ROOT_DIR, "数据", "SST-2")
MODEL_DIR       = os.path.join(ROOT_DIR, "rf_sst2")            # 保存模型
SUBMISSION_FILE = os.path.join(ROOT_DIR, "submission_rf.tsv")  # 提交结果

# 创建输出目录（若不存在）
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------
# 1. 读取本地 TSV
# ------------------------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.tsv"), sep="\t")
dev_df   = pd.read_csv(os.path.join(DATA_DIR, "dev.tsv"),   sep="\t")
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.tsv"),  sep="\t")

print(f"训练样本：{len(train_df)}，验证样本：{len(dev_df)}，测试样本：{len(test_df)}")

# ------------------------------------------------
# 2. 构建 Pipeline：TF-IDF → 随机森林
# ------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True
    )),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    ))
])

# ------------------------------------------------
# 3. 训练
# ------------------------------------------------
X_train, y_train = train_df["sentence"], train_df["label"]
X_dev,   y_dev   = dev_df["sentence"],   dev_df["label"]

print("开始训练随机森林...")
pipeline.fit(X_train, y_train)
print("训练完成！")

# ------------------------------------------------
# 4. 验证集评估
# ------------------------------------------------
y_pred = pipeline.predict(X_dev)
acc = accuracy_score(y_dev, y_pred)
print(f"验证集准确率：{acc:.4f}")
print("详细分类报告：\n", classification_report(y_dev, y_pred, target_names=["负向", "正向"]))

# ------------------------------------------------
# 5. 保存模型
# ------------------------------------------------
joblib.dump(pipeline, os.path.join(MODEL_DIR, "rf_model.pkl"))
print("模型已保存到：", MODEL_DIR)

# ------------------------------------------------
# 6. 测试集推理
# ------------------------------------------------
if "label" not in test_df.columns:
    # 官方测试集无标签
    preds = pipeline.predict(test_df["sentence"])
    submission = pd.DataFrame({"index": test_df["index"], "prediction": preds})
    submission.to_csv(SUBMISSION_FILE, sep="\t", index=False)
    print("提交文件已生成：", SUBMISSION_FILE)
else:
    # 本地带标签测试集
    preds = pipeline.predict(test_df["sentence"])
    test_acc = accuracy_score(test_df["label"], preds)
    print("测试集准确率：", test_acc)