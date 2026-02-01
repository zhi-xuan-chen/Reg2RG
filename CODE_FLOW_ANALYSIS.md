# 📘 Phân Tích Luồng Code Reg2RG - Chi Tiết Từng Bước

## 🎯 Tổng Quan

**Reg2RG** (Region-to-Report Generation) là một model Vision-Language kết hợp:
- **3D Vision Encoder** (ViT-3D) để trích xuất đặc trưng hình ảnh từ CT scans
- **LLaMA Language Model** để tạo báo cáo y khoa
- **Xử lý đa vùng** để xử lý nhiều vùng giải phẫu khác nhau

---

## 📂 Cấu Trúc Project

```
Reg2RG/
├── src/
│   ├── train_radgenome.py        # [ĐIỂM BẮT ĐẦU] - Script training chính
│   ├── test_radgenome.py         # Script testing/inference
│   ├── Model/
│   │   ├── Reg2RG.py            # [MODEL CHÍNH] - Kiến trúc model
│   │   ├── my_embedding_layer.py # Lớp embedding Vision-Language
│   │   ├── vit_3d.py            # 3D Vision Transformer
│   │   ├── helpers.py           # Perceiver Resampler
│   │   └── ...
│   └── Dataset/
│       ├── radgenome_dataset_train.py  # Dataset loader cho training
│       └── ...
├── configs/                      # File cấu hình
├── checkpoints/                  # Model checkpoints
└── evaluation/                   # Scripts đánh giá
```

---

## 🚀 Luồng Xử Lý Chính

### MỨC 1: Điểm Bắt Đầu

```python
# File: train_radgenome.py
main()
  ├─ set_seed(42)                      # Cố định random seed
  ├─ parser.parse_args()               # Parse tham số dòng lệnh
  ├─ RadGenomeDataset_Train(...)      # Tạo dataset
  ├─ Reg2RG(...)                      # Khởi tạo model
  └─ Trainer.train()                  # Bắt đầu training
```

**Chi tiết từng bước:**

#### 1. `set_seed(42)` - Cố Định Random Seed
```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
**Mục đích:** Đảm bảo kết quả có thể tái tạo (chạy lại cho cùng kết quả)

#### 2. `parser.parse_args()` - Đọc Tham Số
```python
parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```

**Các tham số bao gồm:**
- **ModelArguments:**
  - `tokenizer_path`: Đường dẫn tokenizer (LLaMA)
  - `lang_encoder_path`: Đường dẫn language model
  - `pretrained_visual_encoder`: Đường dẫn ViT-3D pretrained
  - `pretrained_adapter`: Đường dẫn adapter weights

- **DataArguments:**
  - `data_folder`: Thư mục chứa CT scans (.nii.gz)
  - `mask_folder`: Thư mục chứa segmentation masks
  - `report_file`: File CSV chứa báo cáo

- **TrainingArguments:**
  - `batch_size`, `learning_rate`, `num_epochs`, v.v.

#### 3. `RadGenomeDataset_Train(...)` - Tạo Dataset
```python
Train_dataset = RadGenomeDataset_Train(
    text_tokenizer=model_args.tokenizer_path,
    data_folder=data_args.data_folder,
    mask_folder=data_args.mask_folder,
    csv_file=data_args.report_file,
    cache_dir=data_args.monai_cache_dir,
)
```

**Luồng trong Dataset:**
```
RadGenomeDataset_Train.__init__()
  ├─ Load tokenizer
  ├─ Thêm special tokens (<image>, <region>, ...)
  ├─ load_accession_sentences(csv_file)     # Load báo cáo từ CSV
  ├─ prepare_samples()                      # Quét thư mục, tạo danh sách mẫu
  └─ Thiết lập transforms (resize, normalize, ...)
```

#### 4. `Reg2RG(...)` - Khởi Tạo Model
```python
model = Reg2RG(
    lang_model_path=model_args.lang_encoder_path,
    text_tokenizer_path=model_args.tokenizer_path,
    pretrained_visual_encoder=model_args.pretrained_visual_encoder,
    pretrained_adapter=model_args.pretrained_adapter,
)
```

**Luồng trong Model.__init__():**
```
Reg2RG.__init__()
  ├─ Load LlamaTokenizer
  ├─ Thêm special tokens
  │   ├─ <image0>, <image1>, ..., <image31>  (32 image tokens)
  │   └─ <region0>, <region1>, ..., <region329> (10 regions × 33 tokens)
  ├─ Load LlamaForCausalLM
  ├─ Áp dụng LoRA (Low-Rank Adaptation)
  │   └─ Config: r=8, lora_alpha=32, lora_dropout=0.1
  ├─ Khởi tạo lớp MyEmbedding
  │   ├─ ViT-3D encoder (vision_encoder)
  │   ├─ Mask encoder (mask_encoder)
  │   ├─ Perceiver Resampler
  │   └─ Các lớp FC
  └─ Khởi tạo loss_function = CrossEntropyLoss()
```

#### 5. `Trainer.train()` - Bắt Đầu Training
```python
trainer = Trainer(
    model=model,
    train_dataset=Train_dataset,
    args=training_args,
    data_collator=DataCollator(),
)
trainer.train()
```

---

### MỨC 2: Luồng Load Dataset

```python
# File: radgenome_dataset_train.py

__getitem__(index)
  ├─ Load ảnh CT scan (.nii.gz)
  ├─ Load segmentation masks cho các vùng
  ├─ mask_nii_img_to_tensor()
  │   ├─ Load dữ liệu CT: nib.load(img_path).get_fdata()
  │   ├─ Với mỗi vùng:
  │   │   ├─ Load mask: nib.load(mask_path).get_fdata()
  │   │   ├─ Áp dụng mask: masked_img = img * mask
  │   │   ├─ Crop foreground
  │   │   ├─ Resize về (256, 256, 64)
  │   │   ├─ Chuẩn hóa giá trị HU: [-1000, 200] → [0, 1]
  │   │   └─ Lặp lại thành 3 channels
  │   └─ Xử lý ảnh đầy đủ tương tự
  ├─ Tạo text prompt
  │   ├─ Thêm instruction
  │   ├─ Thêm image tokens
  │   └─ Thêm region tokens cho mỗi vùng
  ├─ Tạo combined report (ground truth)
  ├─ Tokenize prompt + report
  └─ Tạo labels (mask phần prompt bằng -100)
```

**Output của `__getitem__()`:**
```python
{
    'lang_x': text_input,              # Text đã tokenize [seq_len]
    'vision_x': mask_img_tensors,      # Dict các vision features
    'mask_x': mask_tensors,            # Dict các mask features  
    'region2area': region2area,        # Mapping region_id → tên anatomy
    'attention_mask': attention_mask,  # Attention mask [seq_len]
    'label': label                     # Target labels [seq_len]
}
```

**Chi tiết `vision_x` và `mask_x`:**
```python
vision_x = {
    'image': tensor [1, 3, 256, 256, 64],    # CT scan đầy đủ
    'lung': tensor [1, 3, 256, 256, 64],     # Vùng phổi đã mask
    'heart': tensor [1, 3, 256, 256, 64],    # Vùng tim đã mask
    ...
}

mask_x = {
    'lung': tensor [1, 256, 256, 64],        # Binary mask
    'heart': tensor [1, 256, 256, 64],
    ...
}
```

---

### MỨC 3: Ghép Dữ Liệu Thành Batch

```python
# File: train_radgenome.py - DataCollator

DataCollator.__call__(instances)
  ├─ Trích xuất các trường từ batch
  ├─ Stack lang_x, attention_mask, labels → Tensors
  ├─ Với mỗi REGION:
  │   ├─ Kiểm tra xem có mẫu nào có vùng này không
  │   ├─ Nếu có: thu thập vision/mask tensors
  │   ├─ Nếu không: thêm placeholders toàn số 0
  │   └─ Loại bỏ các vùng vô dụng (toàn số 0)
  └─ Trả về dict đã batch
```

**Output của DataCollator:**
```python
{
    'lang_x': [B, seq_len],
    'vision_x': {
        'image': [B, 3, 256, 256, 64],
        'lung': [B, 3, 256, 256, 64],
        'heart': [B, 3, 256, 256, 64],
        ...
    },
    'mask_x': {
        'lung': [B, 256, 256, 64],
        'heart': [B, 256, 256, 64],
        ...
    },
    'region2area': [region2area_dict_cho_mỗi_mẫu],
    'attention_mask': [B, seq_len],
    'labels': [B, seq_len]
}
```

---

### MỨC 4: Forward Pass Của Model

```python
# File: Reg2RG.py

Reg2RG.forward(lang_x, vision_x, mask_x, region2area, attention_mask, labels)
  ├─ MyEmbedding.forward()
  │   ├─ Xử lý ảnh đầy đủ
  │   │   ├─ vision_encoder(image)        # ViT-3D encoding
  │   │   ├─ perceiver(features)          # Giảm xuống 32 tokens
  │   │   └─ fc(features)                 # Project lên chiều LLaMA (4096)
  │   ├─ Với mỗi vùng:
  │   │   ├─ vision_encoder(region_image)
  │   │   ├─ perceiver(features)
  │   │   ├─ mask_encoder(mask)           # Encode binary mask
  │   │   └─ Nối [region_feat, mask_feat]  # 32 + 1 = 33 tokens
  │   ├─ Kết hợp embeddings:
  │   │   ├─ Word embeddings (32000 tokens)
  │   │   ├─ Special tokens (<image>, <region>)
  │   │   ├─ Image embeddings (32 tokens)
  │   │   └─ Region embeddings (N_regions × 33 tokens)
  │   └─ Dùng one-hot + matmul để lấy embeddings cuối cùng
  ├─ LlamaForCausalLM.forward()
  │   ├─ Input: embeddings từ MyEmbedding
  │   ├─ Các lớp Transformer (với LoRA)
  │   └─ Output: logits [B, seq_len, vocab_size]
  └─ Tính loss và accuracy
```

---

### MỨC 5: Chi Tiết Lớp Embedding

```python
# File: my_embedding_layer.py

MyEmbedding.forward(vision_x, mask_x, text_input, region2areas)
  
  # Bước 1: Encode ảnh đầy đủ
  B, S, C, H, W, D = vision_x['image'].shape  # [batch, 1, 3, 256, 256, 64]
  image_features = vision_encoder(vision_x['image'])    # → [B, num_patches, 768]
  image_embedding = perceiver(image_features) # → [B, 32, 768]
  image_embedding = fc(image_embedding)       # → [B, 32, 4096]
  
  # Bước 2: Encode từng vùng
  for region_name in vision_x.keys():  # ví dụ: 'lung', 'heart', ...
      region_features = vision_encoder(vision_x[region_name])
      region_embedding = perceiver(region_features)  # → [B, 32, 768]
      region_embedding = fc(region_embedding)        # → [B, 32, 4096]
      
      mask_features = mask_encoder(mask_x[region_name])  # → [B, num_patches, 255]
      mask_embedding = mean(mask_features)  # → [B, 255]
      mask_embedding = mask_fc(mask_embedding)  # → [B, 4096]
      
      # Nối: 32 region tokens + 1 mask token = 33 tokens
      combined = cat([region_embedding, mask_embedding.unsqueeze(1)], dim=1)
      # → [B, 33, 4096]
  
  # Bước 3: Sắp xếp region embeddings theo mapping region2area
  max_regions = len(region2areas[0])  # Số vùng trong batch này
  vision_region_embedding = zeros([B, 33*max_regions, 4096])
  
  for batch_idx in range(B):
      for region_idx in range(len(region2areas[batch_idx])):
          region_name = region2areas[batch_idx][region_idx]
          vision_region_embedding[batch_idx, region_idx*33:(region_idx+1)*33, :] = \
              region_embeddings[region_name][batch_idx, :, :]
  
  # Bước 4: Kết hợp tất cả embeddings
  embedding_weight = cat([
      self.weight,                    # [32000, 4096] - Word embeddings
      self.image_token_weight,        # [2, 4096] - <image>, </image>
      self.region_token_weight,       # [2, 4096] - <region>, </region>
      image_embedding,                # [B, 32, 4096] - Đặc trưng ảnh thực tế
      vision_region_embedding         # [B, 33*N_regions, 4096] - Đặc trưng vùng
  ], dim=1)
  # → [B, 32000+4+32+33*N_regions, 4096]
  
  # Bước 5: Chuyển text_input tokens thành embeddings dùng one-hot + matmul
  text_one_hot = F.one_hot(text_input, num_classes=embedding_weight.shape[1])
  # → [B, seq_len, 32000+4+32+33*N_regions]
  
  output_embedding = matmul(text_one_hot, embedding_weight)
  # → [B, seq_len, 4096]
  
  return output_embedding
```

**Ví dụ cụ thể:**
```
Text đầu vào: "<image><image0><image1>...<image31></image> <region><region0>...<region32></region> Generate report..."

Token IDs:
- "Generate" → 1234 → Dùng word embedding weight[1234]
- "<image0>" → 32000 → Dùng image_embedding[0]
- "<region0>" → 32002 → Dùng vision_region_embedding[0] (lung region token 0)
- "<region33>" → 32035 → Dùng vision_region_embedding[33] (heart region token 0)

Embedding cuối: [B, seq_len, 4096]
```

---

## 💰 Hàm Loss

### 1. **CrossEntropyLoss** (Chính)

**Vị trí:** `Reg2RG.py`, dòng 84 & dòng 99

```python
self.loss_function = nn.CrossEntropyLoss()

output = self.lang_model(
    inputs_embeds=input_embedding,
    attention_mask=attention_mask,
    labels=labels
)
```

**Chi tiết:**
- **Loại:** Language Modeling Loss
- **Công thức:**
  ```
  Loss = -Σ log(P(y_t | y_<t, vision_context))
  ```
  Trong đó:
  - `y_t`: Token ground truth tại vị trí t
  - `y_<t`: Các tokens trước đó
  - `vision_context`: Visual embeddings từ CT scan

- **Label Masking:**
  ```python
  # File: radgenome_dataset_train.py, dòng 324-328
  label = text_input.clone()
  label[label == pad_token_id] = -100        # Bỏ qua padding
  label[label >= voc_size] = -100            # Bỏ qua special tokens
  label[:prompt_length] = -100               # Bỏ qua prompt, chỉ học answer
  ```

- **Cơ chế hoạt động:**
  1. Model dự đoán logits: `[B, seq_len, vocab_size]`
  2. So sánh với ground truth labels: `[B, seq_len]`
  3. Chỉ tính loss trên phần answer (labels != -100)
  4. Tính trung bình loss trên batch và tokens

**Ví dụ:**
```
Input:  <image>... Instruction... <region>... Region tokens...
Label:  -100  -100  -100  -100  -100  -100  -100  -100
        [Bị bỏ qua - phần instruction]

Output: "The region 0 is lung: No mass detected."
Label:  token_id(lung) token_id(:) token_id(No) token_id(mass) ...
        [Học thực sự ở đây]

Loss = CrossEntropy(predicted_logits, ground_truth_tokens)
```

### 2. **LoRA Training** (Ngầm định)

```python
# File: Reg2RG.py, dòng 69-72
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,                    # Rank của ma trận low-rank
    lora_alpha=32,          # Hệ số scaling
    lora_dropout=0.1
)
self.lang_model = get_peft_model(self.lang_model, peft_config)
```

**Không phải hàm loss riêng, nhưng ảnh hưởng training:**
- Chỉ train các LoRA adapters (ma trận low-rank)
- Base LLaMA weights bị đóng băng
- Giảm số parameters cần train: ~8 triệu (thay vì ~7 tỷ)

### 3. **Gradient Checkpointing**

```python
# File: Reg2RG.py, dòng 75-76
self.lang_model.gradient_checkpointing_enable()
self.lang_model.enable_input_require_grads()
```

**Không phải loss, nhưng là kỹ thuật tối ưu:**
- Đổi compute lấy memory
- Tính lại activations trong backward pass
- Cho phép train với batch size lớn hơn

---

## 📊 Tóm Tắt Loss Functions

| Loại Loss | Vị trí | Công thức | Mục đích |
|-----------|--------|-----------|----------|
| **CrossEntropyLoss** | `Reg2RG.py:99` | `-Σ log P(y_t\|context)` | Train language model để tạo báo cáo |
| **Label Masking** | `radgenome_dataset_train.py:324-328` | `labels[prompt] = -100` | Chỉ học phần answer, bỏ qua prompt |
| **LoRA Regularization** | Ngầm trong LoRA | L2 regularization | Ngăn overfitting adapters |

**Chỉ có 1 hàm loss chính:** `CrossEntropyLoss` từ PyTorch

---

## 🏥 Danh Sách Bệnh Lý

### 1. **CONDITIONS** (18 bệnh lý trong Dataset)

```python
# File: radgenome_dataset_train.py, dòng 19-38
CONDITIONS = [
    'Medical material',                      # 1. Vật liệu y tế
    'Arterial wall calcification',           # 2. Vôi hóa thành động mạch
    'Cardiomegaly',                          # 3. Tim to
    'Pericardial effusion',                  # 4. Tràn dịch màng ngoài tim
    'Coronary artery wall calcification',    # 5. Vôi hóa động mạch vành
    'Hiatal hernia',                         # 6. Thoát vị thực quản
    'Lymphadenopathy',                       # 7. Hạch to
    'Emphysema',                             # 8. Khí phế thũng
    'Atelectasis',                           # 9. Xẹp phổi
    'Lung nodule',                           # 10. Nốt phổi
    'Lung opacity',                          # 11. Đục phổi
    'Pulmonary fibrotic sequela',            # 12. Di chứng xơ phổi
    'Pleural effusion',                      # 13. Tràn dịch màng phổi
    'Mosaic attenuation pattern',            # 14. Hoa văn khảm
    'Peribronchial thickening',              # 15. Dày thành quanh phế quản
    'Consolidation',                         # 16. Đông đặc
    'Bronchiectasis',                        # 17. Giãn phế quản
    'Interlobular septal thickening',        # 18. Dày vách liên tiểu thùy
]
```

**⚠️ LƯU Ý QUAN TRỌNG:** 
- List này được **định nghĩa nhưng KHÔNG được sử dụng** trong code training
- Đây chỉ là **danh sách tham khảo** các findings có thể xuất hiện
- Model **không phân loại 18 classes này**
- Model **tạo báo cáo dạng text tự do**, không ràng buộc phải mention đúng 18 items

### 2. **REGIONS** (10 vùng giải phẫu)

```python
# File: train_radgenome.py, dòng 15-26
REGIONS = [
    'abdomen',                # 1. Bụng
    'bone',                   # 2. Xương
    'breast',                 # 3. Vú
    'esophagus',              # 4. Thực quản
    'heart',                  # 5. Tim
    'lung',                   # 6. Phổi
    'mediastinum',            # 7. Trung thất
    'pleura',                 # 8. Màng phổi
    'thyroid',                # 9. Tuyến giáp
    'trachea and bronchie',   # 10. Khí quản và phế quản
]
```

**Tổng cộng: 10 vùng giải phẫu chính**

### 3. **Segmentation Masks**

Model sử dụng **segmentation masks** để định vị chính xác các vùng:
- Mỗi vùng có 1 binary mask: `[256, 256, 64]`
- Mask = 1 tại voxels thuộc vùng, 0 ở ngoài
- Masks được encode bởi `mask_encoder` thành features 255 chiều

**Ví dụ:**
```python
# Lung mask shape: [1, 256, 256, 64]
# mask[x, y, z] = 1 nếu voxel thuộc phổi
# mask[x, y, z] = 0 nếu không

# Được xử lý thành:
mask_features = mask_encoder(lung_mask)  # → [B, num_patches, 255]
mask_embedding = mean(mask_features, dim=1)  # → [B, 255]
mask_embedding = mask_fc(mask_embedding)  # → [B, 4096]
```

---

## 🔄 Vòng Lặp Training Hoàn Chỉnh

```
1. [Load Dữ Liệu]
   RadGenomeDataset.__getitem__()
   ├─ Load CT scan (.nii.gz)
   ├─ Load masks cho các vùng
   ├─ Chuẩn hóa & transform
   └─ Tokenize text (prompt + report)
   
2. [Ghép Batch]
   DataCollator.__call__()
   ├─ Batch các mẫu
   ├─ Stack tensors
   └─ Loại bỏ vùng rỗng
   
3. [Forward Pass]
   Reg2RG.forward()
   ├─ MyEmbedding.forward()
   │   ├─ Encode ảnh đầy đủ (32 tokens)
   │   ├─ Encode từng vùng (32+1=33 tokens)
   │   └─ Kết hợp với word embeddings
   ├─ LlamaForCausalLM.forward()
   │   ├─ Các lớp Transformer
   │   └─ Output logits [B, seq_len, 32000]
   └─ Tính CrossEntropyLoss
   
4. [Backward Pass]
   loss.backward()
   ├─ Gradient qua LoRA adapters
   ├─ Gradient qua embedding layer
   └─ Gradient qua visual encoder (đóng băng ❌)
   
5. [Tối Ưu]
   optimizer.step()
   └─ Cập nhật LoRA weights & embedding layer
```

---

## 📐 Kích Thước Model & Parameters

### Parameters Có Thể Train:

1. **LoRA Adapters:**
   - Rank r=8, áp dụng cho attention layers
   - ~8 triệu parameters

2. **Embedding Layer:**
   - `MyEmbedding` (vision encoders, perceiver, FC)
   - ~50-100 triệu parameters

3. **Tổng Có Thể Train:**
   - ~60-110 triệu parameters

### Parameters Đóng Băng:

1. **LLaMA Base Model:**
   - ~7 tỷ parameters
   - Hoàn toàn đóng băng

2. **Vision Encoder (ViT-3D):**
   - Load từ RadFM pretrained
   - Đóng băng (dòng 108-109: `param.requires_grad = False`)

---

## 🎯 Các Điểm Chính

### 1. Xử Lý Đa Vùng
- Mỗi mẫu có **nhiều vùng** (lung, heart, v.v.)
- Mỗi vùng được encode riêng biệt → 33 tokens
- Số lượng vùng khác nhau giữa các mẫu

### 2. Kết Hợp Vision-Language
- **Special tokens** làm "cầu nối":
  - `<image0>` → `<image31>`: Map đến image embeddings
  - `<region0>` → `<region329>`: Map đến region embeddings
- Dùng **one-hot + matmul** để chọn embeddings

### 3. Training Hiệu Quả
- **LoRA**: Chỉ train ~1% parameters
- **Gradient Checkpointing**: Đổi compute lấy memory
- **Frozen Vision Encoder**: Tận dụng pretrained RadFM

### 4. Chiến Lược Label Masking
- Chỉ tính loss trên **phần answer**
- Bỏ qua: padding, special tokens, prompt
- Giúp model tập trung vào tạo báo cáo

---

## 📝 Tóm Tắt

**Luồng chính:** Dữ liệu → Vision Encoding → Language Modeling → Loss

**Loss:** Chỉ 1 loss chính = `CrossEntropyLoss` cho language modeling

**Bệnh lý:** 18 conditions được đề cập trong báo cáo (nhưng không dùng cho classification)

**Vùng:** 10 anatomy regions với segmentation masks

**Đặc điểm:** Xử lý đa vùng, Kết hợp Vision-Language, Training hiệu quả bằng LoRA
