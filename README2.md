# Hướng Dẫn Chi Tiết Training Reg2RG

Tài liệu này hướng dẫn chi tiết các bước để train lại model Reg2RG trên bộ dataset RadGenome-ChestCT, với môi trường `reg2rg`.

## 1. Cấu Trúc Thư Mục Thực Tế
Sau khi setup, hệ thống trên máy bạn có cấu trúc **chính xác** như sau:

```
/home/hoangnv/AICD_HA/Reg2RG/
├── blobs/                      # Dữ liệu model Llama-2 (đã di chuyển từ cache)
├── checkpoints/                # Chứa các model pre-trained
│   ├── Llama-2-7b-chat-hf/     # Thư mục config và symlink trỏ về blobs
│   ├── RadFM_vit3d.pth         # Model xử lý ảnh 3D
│   └── RadFM_perceiver_fc.pth  # Adapter
├── src/Dataset/RadGenome-ChestCT/dataset/ # Dữ liệu training
│   ├── train_preprocessed/     # Ảnh CT xử lý
│   ├── train_region_mask/      # Mask vùng (đã giải nén)
│   └── radgenome_files/train_region_report.csv # File báo cáo
├── configs/train_radgenome/
│   └── hoangnv.sh              # File cấu hình training (đường dẫn tương đối)
└── scripts/
    └── train_radgenome.sh      # Script chạy chính
```

## 2. Chuẩn Bị Môi Trường
Máy bạn đã có môi trường `reg2rg`. Hãy kích hoạt nó trước khi chạy bất kỳ lệnh nào:

```bash
conda activate reg2rg
```

Kiểm tra xem `torchrun` đã sẵn sàng chưa:
```bash
which torchrun
# Kết quả nên trả về đường dẫn, ví dụ: .../envs/reg2rg/bin/torchrun
```

## 3. Lệnh Chạy Training
Quy trình chuẩn để chạy train:

**Bước 1:** Di chuyển vào thư mục chứa script
```bash
cd /home/hoangnv/AICD_HA/Reg2RG/scripts
```

**Bước 2:** Chạy lệnh train với config `hoangnv`
```bash
bash train_radgenome.sh hoangnv
```

## 4. Giải Thích File Cấu Hình (`hoangnv.sh`)
File `configs/train_radgenome/hoangnv.sh` đã được thiết lập dùng **đường dẫn tương đối** (`../`) để linh hoạt:
-   **Model**: Trỏ về `../checkpoints/...`
-   **Data**: Trỏ về `../src/Dataset/...`
-   **Output**: Lưu tại `../outputs/Reg2RG_radgenome`

Điều này giúp bạn có thể chạy lệnh từ thư mục `scripts` mà không lo bị sai đường dẫn tuyệt đối khi di chuyển code.

## 5. Xử Lý Sự Cố
-   Nếu gặp lỗi `OutOfMemory`: Mở file `hoangnv.sh`, giảm `gradient_accumulation_steps` (hiện là 8) hoặc đảm bảo không có tiến trình nào khác chiếm GPU.
-   Nếu muốn tải lại Model Llama-2: Xóa thư mục `checkpoints/Llama-2-7b-chat-hf` và `blobs`, sau đó tải lại bằng `hf download`.
