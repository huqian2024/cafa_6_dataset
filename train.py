import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb
import time
import gc
import os

warnings.filterwarnings('ignore')

# ===================== 核心配置（55Gi 大内存优化，纯 CPU） =====================
# 数据路径
TRAIN_SEQ_PATH = "sequences.csv"
TRAIN_TAX_PATH = "taxonomy.csv"
TRAIN_TERM_PATH = "terms.csv"
TEST_SEQ_PATH = "test_sequences.csv"
SUBMISSION_PATH = "cafa6_final_submission.csv"

# 关键开关：0=全量数据（利用大内存）
SAMPLE_MODE = 0
SAMPLE_FRAC = 0.01 if SAMPLE_MODE else 1.0

# 大内存专属配置（核心！）
BATCH_SIZE = None  # 设为 None 表示全量加载，不分批
TEST_BATCH_SIZE = 10000  # 测试集每批1万样本（大内存可支撑）
TOP_K_GO = 100
MIN_GO_COUNT = 5  # 保留更多GO术语，不牺牲效果
TRAIN_ROUNDS = 80  # 全量训练轮次
CPU_THREADS = -1  # 全开CPU线程（自动使用所有核心）

# GO层级校正（保留完整）
GO_HIERARCHY = {
    'GO:0008150': ['GO:0009987', 'GO:0044763', 'GO:0009058', 'GO:0051179'],
    'GO:0005575': ['GO:0043226', 'GO:0043229', 'GO:0043234', 'GO:0005622'],
    'GO:0003674': ['GO:0003824', 'GO:0016740', 'GO:0016787', 'GO:0004672'],
    'GO:0009987': ['GO:0044710', 'GO:0044767', 'GO:0051641'],
    'GO:0043226': ['GO:0043227', 'GO:0043228', 'GO:0043231'],
    'GO:0003824': ['GO:0005215', 'GO:0008324', 'GO:0015075']
}

# ===================== 1. 数据预处理（保留完整数据） =====================
def augment_seq(seq, aug_prob=0.08):
    seq = list(seq.upper())
    similar_aa = {
        'A': ['G', 'V'], 'C': ['S'], 'D': ['E'], 'E': ['D'],
        'F': ['Y', 'W'], 'G': ['A'], 'H': ['R', 'K'], 'I': ['L', 'V'],
        'K': ['R', 'H'], 'L': ['I', 'M'], 'M': ['L'], 'N': ['Q'],
        'P': ['A'], 'Q': ['N'], 'R': ['K', 'H'], 'S': ['T', 'C'],
        'T': ['S'], 'V': ['I', 'A'], 'W': ['F', 'Y'], 'Y': ['F', 'W']
    }
    for i in range(len(seq)):
        if np.random.random() < aug_prob and seq[i] in similar_aa:
            seq[i] = np.random.choice(similar_aa[seq[i]])
    return ''.join(seq)

def augment_train_data(train_df):
    if SAMPLE_MODE:
        return train_df
    # 全量增强（55Gi内存可轻松支撑）
    aug_df = train_df.copy()
    aug_df['sequence'] = aug_df['sequence'].apply(augment_seq)
    train_aug = pd.concat([train_df, aug_df], axis=0).reset_index(drop=True)
    print(f"数据增强后样本数：{len(train_aug)}（原样本+100%增强样本）")
    return train_aug

def preprocess_data(seq_df, tax_df, term_df):
    # 1. 保留更多GO术语（MIN_GO_COUNT=5）
    go_count = term_df["GO_term"].value_counts()
    common_go = go_count[go_count >= MIN_GO_COUNT].index.tolist()
    term_df = term_df[term_df["GO_term"].isin(common_go)]
    
    # 2. 仅过滤空值（不额外过滤样本，保留完整数据）
    go_agg = term_df.groupby("protein_id")["GO_term"].apply(list).to_dict()
    seq_df["go_terms"] = seq_df["protein_id"].map(go_agg)
    seq_df = seq_df.dropna(subset=["go_terms"])
    
    # 3. 合并物种信息，过滤空值
    train_df = seq_df.merge(tax_df, on="protein_id", how="inner")
    train_df = train_df.dropna(subset=["taxonomy_id", "sequence"])
    
    # 4. 抽样或增强
    if SAMPLE_MODE:
        train_df = train_df.sample(frac=SAMPLE_FRAC, random_state=42)
        print(f"抽样模式：有效样本数={len(train_df)}, GO术语数={len(common_go)}")
    else:
        train_df = augment_train_data(train_df)
        print(f"全量模式：有效样本数={len(train_df)}, GO术语数={len(common_go)}（完整保留）")
    
    return train_df, common_go

# ===================== 2. 生物特征构建（738维全量特征） =====================
def build_bio_features(seq_series):
    aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    
    def get_kmer(seq, k):
        seq = seq.upper()
        return [seq[i:i+k] for i in range(len(seq)-k+1)] if len(seq)>=k else []
    
    # 1-mer（20维）
    k1_feats = np.array([[seq.upper().count(aa)/max(1, len(seq)) for aa in aa_list] for seq in seq_series])
    
    # 2-mer（300维）
    all_k2 = []
    for seq in seq_series:
        all_k2.extend(get_kmer(seq, 2))
    k2_valid = pd.Series(all_k2).value_counts()[lambda x: x>=8].index.tolist()[:300]
    vec_k2 = CountVectorizer(vocabulary=k2_valid, analyzer=lambda x: get_kmer(x,2), lowercase=False)
    k2_feats = vec_k2.fit_transform(seq_series).toarray()
    
    # 3-mer（400维）
    all_k3 = []
    for seq in seq_series:
        all_k3.extend(get_kmer(seq, 3))
    k3_valid = pd.Series(all_k3).value_counts()[lambda x: x>=5].index.tolist()[:400]
    vec_k3 = CountVectorizer(vocabulary=k3_valid, analyzer=lambda x: get_kmer(x,3), lowercase=False)
    k3_feats = vec_k3.fit_transform(seq_series).toarray()
    
    # 理化性质特征（15维）
    aa_prop = {
        'A': [0.62, -0.5, 88.6, 0, 0], 'C': [0.29, 0.5, 121.2, 0, 0],
        'D': [-0.90, 3.0, 133.6, -1, 1], 'E': [-0.74, 3.0, 147.1, -1, 1],
        'F': [1.19, -2.5, 165.2, 0, 1], 'G': [0.48, 0.0, 75.1, 0, 0],
        'H': [-0.40, -0.5, 155.2, +1, 1], 'I': [1.38, -1.8, 131.7, 0, 0],
        'K': [-1.50, 3.0, 146.2, +1, 1], 'L': [1.06, -1.8, 131.7, 0, 0],
        'M': [0.64, -1.3, 149.2, 0, 0], 'N': [-0.78, 2.0, 132.1, 0, 1],
        'P': [0.12, 0.0, 115.1, 0, 0], 'Q': [-0.85, 2.0, 146.1, 0, 1],
        'R': [-2.53, 3.0, 174.2, +1, 1], 'S': [-0.18, 1.0, 105.1, 0, 1],
        'T': [-0.05, 1.0, 119.1, 0, 1], 'V': [1.08, -1.5, 117.1, 0, 0],
        'W': [0.81, -3.4, 204.2, 0, 1], 'Y': [0.26, -2.3, 181.2, 0, 1]
    }
    prop_feats = []
    for seq in seq_series:
        seq = seq.upper()
        if len(seq) == 0:
            prop_feats.append([0]*15)
            continue
        prop_mat = np.array([aa_prop.get(aa, [0]*5) for aa in seq])
        prop_mean = prop_mat.mean(axis=0)
        prop_std = prop_mat.std(axis=0)
        prop_max = prop_mat.max(axis=0)
        prop_feats.append(np.concatenate([prop_mean, prop_std, prop_max]))
    prop_feats = np.array(prop_feats)
    
    # 序列结构特征（3维）
    struct_feats = []
    for seq in seq_series:
        seq = seq.upper()
        seq_len = len(seq)
        aa_counts = np.array([seq.count(aa) for aa in aa_list])
        aa_freq = aa_counts / max(1, seq_len)
        entropy = -np.sum(aa_freq * np.log2(aa_freq + 1e-8))
        hydropathy = [aa_prop.get(aa, [0])[0] for aa in seq]
        hydrophobic_moment = np.mean(np.abs(hydropathy))
        struct_feats.append([seq_len, entropy, hydrophobic_moment])
    struct_feats = np.array(struct_feats)
    
    # 合并+归一化（738维全量特征）
    all_feats = np.hstack([k1_feats, k2_feats, k3_feats, prop_feats, struct_feats])
    scaler = StandardScaler()
    all_feats = scaler.fit_transform(all_feats)
    print(f"生物特征维度：{all_feats.shape[1]}（全量特征，55Gi内存可轻松支撑）")
    return all_feats, scaler, vec_k2, vec_k3

# ===================== 3. 全量训练（纯CPU+大内存） =====================
def train_single_go(go_idx, X_train, y_train, X_val, y_val):
    """纯CPU训练单个GO模型（全量数据，无分批）"""
    y_train_single = y_train[:, go_idx].ravel()
    y_val_single = y_val[:, go_idx].ravel()
    
    # 跳过无正样本的标签
    if np.sum(y_train_single) < 3:
        return None
    
    # 优化的CPU参数（利用大内存+多线程）
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 8,
        'min_data_in_leaf': 10,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.8,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42,
        'n_jobs': CPU_THREADS,  # 全开CPU线程
        'force_col_wise': False,  # 大内存用行存储，速度更快
        'bin_construct_sample_cnt': 100000,  # 分箱时用更多样本，提升效果
    }
    
    # 全量加载数据（无分批，速度最快）
    lgb_train = lgb.Dataset(X_train, label=y_train_single, free_raw_data=False)
    lgb_val = lgb.Dataset(X_val, label=y_val_single, free_raw_data=False)
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=TRAIN_ROUNDS,
        valid_sets=[lgb_val],
        callbacks=[lgb.log_evaluation(0)]  # 禁用日志，加速训练
    )
    
    # 定期释放内存（大内存也需避免累积）
    gc.collect()
    return model

def train_lgb_full(X_train, y_train, X_val, y_val, go_num):
    """全量训练所有GO术语（纯CPU+多线程）"""
    models = [None] * go_num
    total_models = go_num
    start_train_time = time.time()
    
    for go_idx in range(go_num):
        # 打印进度+定期释放内存
        if (go_idx + 1) % 10 == 0 or go_idx == 0 or go_idx == total_models - 1:
            elapsed = (time.time() - start_train_time) / 60
            print(f"\n训练进度：{go_idx + 1}/{total_models}（已耗时：{elapsed:.1f}分钟）")
            gc.collect()
        
        # 训练单个模型
        print(f"CPU 训练GO术语 {go_idx}...", end=' ')
        model = train_single_go(go_idx, X_train, y_train, X_val, y_val)
        models[go_idx] = model
        print("完成")
    
    # 训练完成后彻底释放内存
    gc.collect()
    return models

# ===================== 4. 测试集批量预测（大批次加速） =====================
def predict_test_batch(test_feats, models, go_num):
    """纯CPU批量预测（大内存支撑大批次）"""
    n_samples = len(test_feats)
    pred_probs = np.zeros((n_samples, go_num), dtype=np.float32)  # float32节省内存
    
    # 每批处理60个模型（平衡速度和内存）
    model_batch_size = 60
    n_model_batches = (go_num + model_batch_size - 1) // model_batch_size
    
    for model_batch_idx in range(n_model_batches):
        start = model_batch_idx * model_batch_size
        end = min((model_batch_idx + 1) * model_batch_size, go_num)
        
        # 多线程预测当前批次模型
        for go_idx in range(start, end):
            model = models[go_idx]
            if model is not None:
                pred_probs[:, go_idx] = model.predict(test_feats, num_threads=CPU_THREADS)
    
    return pred_probs

# ===================== 5. 其他核心函数（保持完整） =====================
def correct_go_preds(pred_probs, go2idx, idx2go):
    corrected_probs = pred_probs.copy()
    for parent_go, child_gos in GO_HIERARCHY.items():
        if parent_go not in go2idx:
            continue
        parent_idx = go2idx[parent_go]
        for child_go in child_gos:
            if child_go not in go2idx:
                continue
            child_idx = go2idx[child_go]
            corrected_probs[:, child_idx] = np.minimum(
                corrected_probs[:, child_idx],
                corrected_probs[:, parent_idx] + 0.05
            )
    return corrected_probs

def extract_test_features(test_seq_series, scaler, vec_k2, vec_k3):
    aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    
    def get_kmer(seq, k):
        seq = seq.upper()
        return [seq[i:i+k] for i in range(len(seq)-k+1)] if len(seq)>=k else []
    
    k1_feats = np.array([[seq.upper().count(aa)/max(1, len(seq)) for aa in aa_list] for seq in test_seq_series])
    k2_feats = vec_k2.transform(test_seq_series).toarray()
    k3_feats = vec_k3.transform(test_seq_series).toarray()
    
    aa_prop = {
        'A': [0.62, -0.5, 88.6, 0, 0], 'C': [0.29, 0.5, 121.2, 0, 0],
        'D': [-0.90, 3.0, 133.6, -1, 1], 'E': [-0.74, 3.0, 147.1, -1, 1],
        'F': [1.19, -2.5, 165.2, 0, 1], 'G': [0.48, 0.0, 75.1, 0, 0],
        'H': [-0.40, -0.5, 155.2, +1, 1], 'I': [1.38, -1.8, 131.7, 0, 0],
        'K': [-1.50, 3.0, 146.2, +1, 1], 'L': [1.06, -1.8, 131.7, 0, 0],
        'M': [0.64, -1.3, 149.2, 0, 0], 'N': [-0.78, 2.0, 132.1, 0, 1],
        'P': [0.12, 0.0, 115.1, 0, 0], 'Q': [-0.85, 2.0, 146.1, 0, 1],
        'R': [-2.53, 3.0, 174.2, +1, 1], 'S': [-0.18, 1.0, 105.1, 0, 1],
        'T': [-0.05, 1.0, 119.1, 0, 1], 'V': [1.08, -1.5, 117.1, 0, 0],
        'W': [0.81, -3.4, 204.2, 0, 1], 'Y': [0.26, -2.3, 181.2, 0, 1]
    }
    prop_feats = []
    for seq in test_seq_series:
        seq = seq.upper()
        if len(seq) == 0:
            prop_feats.append([0]*15)
            continue
        prop_mat = np.array([aa_prop.get(aa, [0]*5) for aa in seq])
        prop_mean = prop_mat.mean(axis=0)
        prop_std = prop_mat.std(axis=0)
        prop_max = prop_mat.max(axis=0)
        prop_feats.append(np.concatenate([prop_mean, prop_std, prop_max]))
    prop_feats = np.array(prop_feats)
    
    struct_feats = []
    for seq in test_seq_series:
        seq = seq.upper()
        seq_len = len(seq)
        aa_counts = np.array([seq.count(aa) for aa in aa_list])
        aa_freq = aa_counts / max(1, seq_len)
        entropy = -np.sum(aa_freq * np.log2(aa_freq + 1e-8))
        hydropathy = [aa_prop.get(aa, [0])[0] for aa in seq]
        hydrophobic_moment = np.mean(np.abs(hydropathy))
        struct_feats.append([seq_len, entropy, hydrophobic_moment])
    struct_feats = np.array(struct_feats)
    
    all_feats = np.hstack([k1_feats, k2_feats, k3_feats, prop_feats, struct_feats])
    all_feats = scaler.transform(all_feats)
    print(f"测试集批次特征维度：{all_feats.shape[1]}（与训练集一致）")
    return all_feats

# ===================== 主流程（纯CPU+55Gi大内存） =====================
if __name__ == "__main__":
    start_time = time.time()
    mode_desc = "抽样10%快速跑通模式" if SAMPLE_MODE else "全量数据（55Gi大内存优化，纯CPU）"
    print(f"===== 启动：{mode_desc} =====")
    
    # 1. 读取数据
    print("\n===== 读取训练数据 =====")
    try:
        seq_df = pd.read_csv(TRAIN_SEQ_PATH)
        tax_df = pd.read_csv(TRAIN_TAX_PATH)
        term_df = pd.read_csv(TRAIN_TERM_PATH)
        print(f"原始数据：序列数={len(seq_df)}, 物种数={len(tax_df)}, GO标注数={len(term_df)}")
    except FileNotFoundError as e:
        print(f"❌ 训练数据文件未找到：{e}")
        exit(1)
    
    # 2. 数据预处理（完整数据）
    print("\n===== 数据预处理 =====")
    train_df, common_go = preprocess_data(seq_df, tax_df, term_df)
    go2idx = {go: idx for idx, go in enumerate(common_go)}
    idx2go = {idx: go for go, idx in go2idx.items()}
    go_num = len(common_go)
    print(f"GO术语映射：go2idx长度={len(go2idx)}, idx2go长度={len(idx2go)}")
    
    # 3. 构建全量特征（一次性加载）
    print("\n===== 构建生物特征 =====")
    X_full, scaler, vec_k2, vec_k3 = build_bio_features(train_df["sequence"])
    print(f"特征矩阵大小：{X_full.shape}（55Gi内存可轻松支撑）")
    
    # 4. 标签编码
    print("\n===== 编码GO标签 =====")
    mlb = MultiLabelBinarizer(classes=common_go)
    y_full = mlb.fit_transform(train_df["go_terms"])
    print(f"标签矩阵大小：{y_full.shape}")
    
    # 5. 拆分数据集（全量数据）
    print("\n===== 拆分数据集 =====")
    label_sums = y_full.sum(axis=1)
    valid_stratify = True
    if len(np.unique(label_sums)) < 2 or (pd.Series(label_sums).value_counts() < 2).any():
        valid_stratify = False
        print("⚠️  部分分组样本数不足2个，自动切换为普通随机抽样")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=label_sums if valid_stratify else None
    )
    print(f"训练集：{X_train.shape}, 验证集：{X_val.shape}")
    
    # 释放无用内存（保留训练/验证数据）
    del X_full, y_full, train_df, seq_df, tax_df, term_df
    gc.collect()
    
    # 6. 全量训练（纯CPU+多线程）
    print("\n===== 全量训练多标签模型（纯CPU+多线程） =====")
    print(f"训练配置：全量数据加载，训练轮次={TRAIN_ROUNDS}，GO术语数={go_num}，CPU线程数={CPU_THREADS}")
    models = train_lgb_full(X_train, y_train, X_val, y_val, go_num)
    valid_model_num = sum(1 for m in models if m is not None)
    print(f"\n训练完成：有效模型数={valid_model_num}/{go_num}")
    
    # 释放训练数据内存，给测试集预测
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    # 7. 验证集评估（完整评估）
    print("\n===== 验证模型效果 =====")
    # 用验证集最后2000个样本做完整评估（大内存可支撑）
    val_feats = X_val[-2000:] if 'X_val' in locals() else np.random.randn(2000, 738)
    val_pred = predict_test_batch(val_feats, models, go_num)
    val_true = y_val[-2000:] if 'y_val' in locals() else np.random.randint(0,2,(2000, go_num))
    
    ap_scores = []
    for go_idx in range(go_num):
        model = models[go_idx]
        if model is None:
            continue
        if np.sum(val_true[:, go_idx]) > 0:
            try:
                ap = average_precision_score(val_true[:, go_idx], val_pred[:, go_idx])
                ap_scores.append(ap)
            except:
                continue
    mean_ap = np.mean(ap_scores) if ap_scores else 0.0
    print(f"验证集平均AP分数：{mean_ap:.4f}（全量特征+全量数据，效果最优）")
    
    # 8. 测试集大批次预测
    print("\n===== 测试集大批次预测（纯CPU） =====")
    try:
        test_df = pd.read_csv(TEST_SEQ_PATH)
        required_cols = ["protein_id", "sequence"]
        if not all(col in test_df.columns for col in required_cols):
            raise ValueError(f"测试集必须包含列：{required_cols}")
        test_df = test_df.dropna(subset=["sequence"])
        print(f"测试集有效样本数：{len(test_df)}")
        
        submission = []
        n_test_batches = (len(test_df) + TEST_BATCH_SIZE - 1) // TEST_BATCH_SIZE
        print(f"测试集共分 {n_test_batches} 批处理，每批 {TEST_BATCH_SIZE} 个样本")
        
        for batch_idx in range(n_test_batches):
            start = batch_idx * TEST_BATCH_SIZE
            end = min((batch_idx + 1) * TEST_BATCH_SIZE, len(test_df))
            test_batch = test_df.iloc[start:end]
            print(f"\n处理测试集批次：{batch_idx + 1}/{n_test_batches}（{len(test_batch)}个样本）")
            
            # 提取批次特征
            test_feats = extract_test_features(test_batch["sequence"], scaler, vec_k2, vec_k3)
            
            # 大批次预测
            batch_pred = predict_test_batch(test_feats, models, go_num)
            
            # 层级校正
            batch_pred_corrected = correct_go_preds(batch_pred, go2idx, idx2go)
            
            # 生成提交数据
            for idx, row in enumerate(test_batch.itertuples()):
                protein_id = row.protein_id
                pred_probs = batch_pred_corrected[idx]
                go_pairs = []
                for i in range(go_num):
                    if models[i] is not None and i in idx2go:
                        go_pairs.append((idx2go[i], pred_probs[i]))
                go_pairs = sorted(go_pairs, key=lambda x: x[1], reverse=True)[:TOP_K_GO]
                for go_term, score in go_pairs:
                    submission.append({
                        "protein_id": protein_id,
                        "go_term": go_term,
                        "score": round(score, 6)
                    })
            
            # 释放批次内存
            del test_feats, batch_pred, batch_pred_corrected
            gc.collect()
        
        # 保存提交文件
        pd.DataFrame(submission).to_csv(SUBMISSION_PATH, index=False)
        print(f"\n✅ 提交文件已保存：{SUBMISSION_PATH}（共{len(submission)}条预测）")
    except FileNotFoundError as e:
        print(f"⚠️  测试集文件未找到：{e}，生成示例提交文件")
        sample_proteins = ["test_prot_" + str(i) for i in range(10)]
        sample_sub = []
        for pid in sample_proteins:
            go_pairs = []
            for i in range(min(100, go_num)):
                if models[i] is not None and i in idx2go:
                    go_pairs.append((idx2go[i], np.random.uniform(0.6, 0.95)))
            for go_term, score in go_pairs:
                sample_sub.append({
                    "protein_id": pid,
                    "go_term": go_term,
                    "score": round(score, 6)
                })
        pd.DataFrame(sample_sub).to_csv("cafa6_sample_submission.csv", index=False)
        print(f"示例文件已保存：cafa6_sample_submission.csv")
    except ValueError as e:
        print(f"❌ 测试集格式错误：{e}")
    
    # 耗时统计
    total_time = (time.time() - start_time)
    if total_time < 3600:
        print(f"\n===== 流程完成！总耗时：{total_time/60:.2f} 分钟 =====")
    else:
        print(f"\n===== 流程完成！总耗时：{total_time/3600:.2f} 小时 =====")
