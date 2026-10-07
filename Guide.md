# Hướng dẫn train robot G1 bằng `mpopi_train`

Hướng dẫn này dành cho người **chưa từng dùng dự án**. Làm lần lượt từng bước, chép nguyên lệnh
vào terminal và nhấn Enter.

Mục tiêu: dạy robot hình người **Unitree G1** (trong mô phỏng) đi theo lệnh vận tốc, tới
**1,5 m/s**, trong **2000 vòng học**. Có 4 phương pháp để so sánh:

| Phương pháp | Tên task dùng trong lệnh | Ý tưởng ngắn gọn |
|---|---|---|
| PPO | `Mpopi-G1-2k-PPO` | Thuật toán gốc, dùng làm mốc so sánh |
| Replay-IS | `Mpopi-G1-2k-Replay-IS` | Dùng lại dữ liệu của 4 vòng trước, có hiệu chỉnh |
| DAgger | `Mpopi-G1-2k-DAgger` | Một bộ điều khiển MPC làm "thầy", chỉ cho robot hành động tốt |
| Replay-IS + DAgger | `Mpopi-G1-2k-Replay-IS-DAgger` | Kết hợp cả hai (phương pháp chính của dự án) |

Có hai cách chạy:

- **Cách A: trên máy của bạn**, nếu máy có card đồ họa NVIDIA (mục 1–7).
- **Cách B: trên Kaggle**, miễn phí, không cần card đồ họa (mục 8).

Không chắc máy mình có card NVIDIA hay không thì làm bước 1.1. Nếu không có, dùng Cách B.

---

## 1. Kiểm tra máy

### 1.1. Card đồ họa NVIDIA

Mở terminal (Ubuntu: nhấn `Ctrl + Alt + T`) và gõ:

```bash
nvidia-smi
```

- Hiện ra một bảng có tên card (ví dụ `Tesla T4`, `RTX 3090`) → **có GPU**, làm tiếp.
- Báo `command not found` → máy không có card NVIDIA hoặc chưa cài driver. Hãy dùng
  **Cách B (Kaggle)** ở mục 8.

Nên có **ít nhất 12 GB bộ nhớ GPU** (cột `Memory` trong bảng) để chạy 4096 robot cùng lúc.

### 1.2. Hệ điều hành

Hướng dẫn viết cho **Linux (Ubuntu)**. Trên Windows nên dùng WSL2 (Ubuntu trong Windows).

---

## 2. Cài công cụ (chỉ làm một lần)

### 2.1. Git (để tải code)

```bash
sudo apt update && sudo apt install -y git curl
```

Lệnh sẽ hỏi mật khẩu máy tính của bạn. Khi gõ, mật khẩu không hiện ra; cứ gõ xong rồi nhấn Enter.

### 2.2. uv (để cài Python và thư viện)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sau đó **đóng terminal và mở lại**, rồi kiểm tra:

```bash
uv --version
```

Hiện ra số phiên bản (ví dụ `uv 0.9.x`) là được. Không cần tự cài Python, `uv` sẽ lo.

---

## 3. Tải code về (chỉ làm một lần)

```bash
cd ~
git clone --branch mpc-stage1 https://github.com/TamasTran/mjlab_MPOPI.git
cd mjlab_MPOPI
```

**Quan trọng:** phải có `--branch mpc-stage1`. Nhánh mặc định (`main`) không có phần code này.

Từ giờ, **mọi lệnh đều chạy trong thư mục `~/mjlab_MPOPI`**. Mỗi lần mở terminal mới, gõ trước:

```bash
cd ~/mjlab_MPOPI
```

---

## 4. Cài thư viện và kiểm tra GPU (chỉ làm một lần)

```bash
uv sync --extra cu128
```

Lệnh này tải khoảng vài GB (PyTorch, CUDA...), mất **5–15 phút** tùy mạng. Sau đó kiểm tra:

```bash
uv run --extra cu128 python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Phải thấy `GPU: True` và tên card. Nếu thấy `GPU: False`, xem mục 9.

---

## 5. Train

### 5.1. Lệnh train cơ bản

Ví dụ train phương pháp chính, seed 1:

```bash
uv run --extra cu128 mpopi-train Mpopi-G1-2k-Replay-IS-DAgger --agent.logger tensorboard --agent.seed 1 --agent.run-name Replay-IS-DAgger_s1
```

Ý nghĩa từng phần:

| Phần | Ý nghĩa |
|---|---|
| `uv run --extra cu128` | Chạy bằng môi trường đã cài, dùng GPU |
| `mpopi-train` | Lệnh train của dự án |
| `Mpopi-G1-2k-Replay-IS-DAgger` | Phương pháp (xem bảng ở đầu trang) |
| `--agent.logger tensorboard` | Lưu biểu đồ trên máy (không cần tài khoản gì) |
| `--agent.seed 1` | "Hạt giống" ngẫu nhiên. Đổi số này để chạy lại thí nghiệm theo cách khác |
| `--agent.run-name ...` | Tên thư mục kết quả, để dễ tìm |

**Đang chạy thì trông như thế nào?** Terminal in ra liên tục các khối `Learning iteration 15/2000`
kèm các con số. Dòng `ETA` cho biết còn bao lâu.

**Mất bao lâu?** Trên GPU T4: PPO khoảng 1,5 giờ, Replay-IS + DAgger khoảng 2 giờ. GPU mạnh hơn thì
nhanh hơn. **Không tắt terminal và không cho máy ngủ** trong lúc train.

**Muốn dừng giữa chừng:** nhấn `Ctrl + C`.

### 5.2. Train các phương pháp khác

Chỉ đổi tên task và tên lần chạy:

```bash
uv run --extra cu128 mpopi-train Mpopi-G1-2k-PPO --agent.logger tensorboard --agent.seed 1 --agent.run-name PPO_s1
```

```bash
uv run --extra cu128 mpopi-train Mpopi-G1-2k-Replay-IS --agent.logger tensorboard --agent.seed 1 --agent.run-name Replay-IS_s1
```

```bash
uv run --extra cu128 mpopi-train Mpopi-G1-2k-DAgger --agent.logger tensorboard --agent.seed 1 --agent.run-name DAgger_s1
```

### 5.3. Train nhiều seed (nên làm khi so sánh)

Kết quả của một lần train có thể lệch khá nhiều so với lần khác. Để so sánh công bằng, mỗi phương
pháp nên chạy **3 seed**. Lệnh sau chạy lần lượt seed 1, 2, 3 (tổng cộng khoảng 3 lần thời gian):

```bash
for s in 1 2 3; do uv run --extra cu128 mpopi-train Mpopi-G1-2k-PPO --agent.logger tensorboard --agent.seed $s --agent.run-name PPO_s$s; done
```

Máy có 2 GPU thì có thể chạy 2 lần train **cùng lúc**, mỗi GPU một lần, ở **hai terminal khác
nhau**. Thêm `CUDA_VISIBLE_DEVICES=0` (terminal 1) hoặc `CUDA_VISIBLE_DEVICES=1` (terminal 2) vào
đầu lệnh:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra cu128 mpopi-train Mpopi-G1-2k-PPO --agent.logger tensorboard --agent.seed 2 --agent.run-name PPO_s2
```

Không chạy 2 lần train trên **cùng một** GPU: chúng sẽ giành nhau và cả hai đều chậm.

---

## 6. Xem kết quả

### 6.1. Kết quả nằm ở đâu

```
logs/rsl_rl/g1_velocity_2k/
└── 2026-10-07_10-00-00_Replay-IS-DAgger_s1/   ← ngày giờ + tên lần chạy
    ├── model_0.pt, model_50.pt, ..., model_1999.pt   ← "bộ não" robot đã học
    ├── events.out.tfevents...                        ← dữ liệu biểu đồ
    └── params/                                       ← cấu hình đã dùng
```

`model_1999.pt` là kết quả cuối cùng.

### 6.2. Xem biểu đồ học

```bash
uv run --extra cu128 tensorboard --logdir logs/rsl_rl/g1_velocity_2k
```

Mở trình duyệt vào địa chỉ **http://localhost:6006**. Các biểu đồ đáng xem:

| Biểu đồ | Ý nghĩa | Tốt khi |
|---|---|---|
| `Episode_Reward/track_linear_velocity` | Robot bám vận tốc tốt đến đâu (tối đa 2,0) | Lên trên 1,5 |
| `Train/mean_episode_length` | Robot đứng được bao lâu trước khi ngã (tối đa 1000) | Gần 1000 |
| `Train/mean_reward` | Tổng điểm thưởng | Tăng dần |

Xem xong thì quay lại terminal, nhấn `Ctrl + C` để tắt TensorBoard.

### 6.3. Đo vận tốc thực tế của robot

Lệnh này cho robot đã học chạy ở các lệnh 0,5 / 1,0 / 1,5 m/s, rồi in ra vận tốc thật và số lần ngã.
Thay đường dẫn bằng thư mục lần chạy của bạn (gõ `ls logs/rsl_rl/g1_velocity_2k/` để xem tên):

```bash
uv run --extra cu128 mpopi-eval --task Mpopi-G1-2k-PPO --controllers policy --num-envs 16 --checkpoint logs/rsl_rl/g1_velocity_2k/TÊN_THƯ_MỤC/model_1999.pt
```

Dùng `--task Mpopi-G1-2k-PPO` cho **mọi** phương pháp: robot và môi trường giống nhau, và cách này
không phải dựng bộ điều khiển MPC khi đánh giá.

Cách đọc kết quả ở lệnh 1,5 m/s:

- Cột `speed` gần 1,5 và cột `|err|` nhỏ (khoảng 0,05) → **bám tốt**.
- Cột `falls/env` bằng 0 → **không ngã**.

### 6.4. Quay video robot

```bash
uv run --extra cu128 mpopi-eval --task Mpopi-G1-2k-PPO --controllers policy --num-envs 1 --speeds 1.5 --checkpoint logs/rsl_rl/g1_velocity_2k/TÊN_THƯ_MỤC/model_1999.pt --video-dir videos
```

Video nằm trong thư mục `videos/`.

### 6.5. Xem robot trực tiếp

```bash
uv run --extra cu128 mpopi-play Mpopi-G1-2k-PPO --checkpoint-file logs/rsl_rl/g1_velocity_2k/TÊN_THƯ_MỤC/model_1999.pt
```

Một cửa sổ (hoặc trang web) mô phỏng sẽ mở ra. Nhấn `Ctrl + C` trong terminal để tắt.

---

## 7. Tùy chỉnh (khi đã quen)

Mọi cấu hình đều đổi được bằng cách thêm vào cuối lệnh train:

| Muốn | Thêm |
|---|---|
| Ít robot hơn (GPU yếu, báo hết bộ nhớ) | `--env.scene.num-envs 2048` |
| Số vòng học khác | `--agent.max-iterations 1000` |
| Ít env phụ cho "thầy" MPC hơn | `--agent.algorithm.mpopi.mpc.num-envs 32` |
| Lưu lên Weights & Biases thay vì máy | Bỏ `--agent.logger tensorboard` (cần đăng nhập W&B trước) |
| Xem mọi tùy chọn | `uv run --extra cu128 mpopi-train Mpopi-G1-2k-PPO --help` |

**Lưu ý:** đổi số robot hoặc số vòng thì kết quả **không còn so sánh trực tiếp được** với các lần
chạy dùng cấu hình chuẩn. Cấu hình chuẩn của từng phương pháp nằm trong
`src/mpopi_train/presets.py`.

---

## 8. Cách B: chạy trên Kaggle (không cần GPU riêng)

Kaggle cho dùng miễn phí 2 GPU T4, khoảng 30 giờ mỗi tuần.

1. Tạo tài khoản ở **https://www.kaggle.com** và **xác minh số điện thoại** (*Settings → Phone
   verification*). Không xác minh thì không bật được Internet và GPU.
2. Tải file notebook **`notebooks/mpc_g1_train_kaggle.ipynb`** về máy. Lấy từ thư mục code đã
   clone, hoặc trên GitHub: mở file, nhấn nút *Download raw file*.
3. Trên Kaggle: **Create → New Notebook**, rồi **File → Import Notebook** và chọn file vừa tải.
4. Ở panel bên phải, mục **Settings**:
   - **Accelerator: GPU T4 x2.** Không chọn P100, vì không chạy được.
   - **Internet: On.**
5. Mở ô **"0. Cấu hình"** nếu muốn chọn phương pháp (`TRAIN`) hoặc seed (`SEEDS`). Mặc định
   notebook train cả 4 phương pháp với seed 1.
6. Nhấn **Save Version** (góc trên bên phải) → chọn **Save & Run All (Commit)** → **Save**.
   Notebook sẽ chạy nền, tối đa 12 giờ. Có thể tắt trình duyệt.
7. Khi chạy xong (trạng thái chuyển thành *Complete*), mở phiên bản đó → tab **Output** → tải file
   **`mpc_g1_train_results.zip`**. Trong đó có biểu đồ, bảng kết quả `ket_qua.txt` và video.

Nếu 12 giờ không đủ cho mọi lần train, notebook tự bỏ qua phần còn thiếu. Để chạy tiếp: tạo phiên
bản mới, thêm output của phiên bản cũ làm *Input* (**Add Input → Your Work**), rồi chạy lại.
Notebook sẽ chỉ train phần còn thiếu.

---

## 9. Gặp lỗi thì làm gì

| Thông báo / hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `nvidia-smi: command not found` | Không có card NVIDIA hoặc chưa cài driver | Dùng Kaggle (mục 8) |
| `GPU: False` ở bước 4 | Thiếu `--extra cu128`, hoặc driver quá cũ | Chạy lại `uv sync --extra cu128`; cập nhật driver NVIDIA |
| `CUDA out of memory` | GPU không đủ bộ nhớ | Thêm `--env.scene.num-envs 2048` (hoặc 1024) |
| `invalid choice: 'Mpopi-G1-2k-...'` hoặc `mpopi-train: command not found` | Sai nhánh hoặc sai thư mục | `cd ~/mjlab_MPOPI` rồi `git checkout mpc-stage1` |
| Hỏi đăng nhập `wandb` | Quên `--agent.logger tensorboard` | Thêm tùy chọn đó vào lệnh |
| Kaggle: không bật được GPU hoặc Internet | Chưa xác minh số điện thoại | Xác minh trong *Settings* của tài khoản Kaggle |
| Kaggle: lỗi CUDA trên P100 | P100 không được hỗ trợ | Chọn **GPU T4 x2** |
| Terminal đứng yên rất lâu ở lần chạy đầu | Đang tải thư viện hoặc biên dịch | Chờ 5–10 phút; chỉ xảy ra lần đầu |

Vẫn không được: chụp **toàn bộ** thông báo lỗi trong terminal (cuộn lên đầu lỗi) và gửi cho người
phụ trách dự án.

---

## 10. Tóm tắt nhanh (khi đã cài xong)

```bash
cd ~/mjlab_MPOPI
uv run --extra cu128 mpopi-train Mpopi-G1-2k-Replay-IS-DAgger --agent.logger tensorboard --agent.seed 1 --agent.run-name Replay-IS-DAgger_s1
uv run --extra cu128 tensorboard --logdir logs/rsl_rl/g1_velocity_2k
uv run --extra cu128 mpopi-eval --task Mpopi-G1-2k-PPO --controllers policy --num-envs 16 --checkpoint logs/rsl_rl/g1_velocity_2k/TÊN_THƯ_MỤC/model_1999.pt
```
