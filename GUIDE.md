# Hướng dẫn train robot G1 bằng `mpopi_train`

Hướng dẫn này dành cho người **chưa từng dùng dự án**. Làm lần lượt từng bước, chép nguyên lệnh
vào terminal và nhấn Enter. Giới thiệu chung về dự án (tiếng Anh) nằm ở [README.md](README.md).

## Dự án làm gì?

Dạy robot hình người **Unitree G1** (trong mô phỏng) đi theo lệnh vận tốc, tới **1,5 m/s**, trong
**2000 vòng học**, và so sánh 4 cách học:

| Phương pháp | Tên task dùng trong lệnh | Ý tưởng ngắn gọn |
|---|---|---|
| PPO | `Mpopi-G1-2k-PPO` | Thuật toán gốc, dùng làm mốc so sánh |
| Replay-IS | `Mpopi-G1-2k-Replay-IS` | Dùng lại dữ liệu của 4 vòng trước, có hiệu chỉnh trọng số |
| DAgger | `Mpopi-G1-2k-DAgger` | Bộ điều khiển MPC làm "thầy", chỉ cho robot hành động tốt ở 110 vòng đầu |
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

Hướng dẫn viết cho **Linux (Ubuntu)**. Trên Windows nên dùng WSL2 (Ubuntu chạy trong Windows).

---

## 2. Cài công cụ (chỉ làm một lần)

### 2.1. Git và curl

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

**Quan trọng:** phải có `--branch mpc-stage1`. Nhánh mặc định (`main`) chỉ có mjlab gốc, không có
phần code của dự án. Đã lỡ clone mà thiếu nhánh thì chạy `git checkout mpc-stage1` trong thư mục.

Từ giờ, **mọi lệnh đều chạy trong thư mục `~/mjlab_MPOPI`**. Mỗi lần mở terminal mới, gõ trước:

```bash
cd ~/mjlab_MPOPI
```

Lấy bản code mới nhất (khi được báo có cập nhật):

```bash
git pull
```

---

## 4. Cài thư viện và kiểm tra GPU (chỉ làm một lần)

```bash
uv sync --extra cu128
```

Lệnh này tải vài GB (PyTorch, CUDA...), mất **5–15 phút** tùy mạng. Sau đó kiểm tra:

```bash
uv run --extra cu128 python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Phải thấy `GPU: True` và tên card. Nếu thấy `GPU: False`, xem mục 9.

**Lưu ý:** mọi lệnh trong hướng dẫn đều bắt đầu bằng `uv run --extra cu128`. Thiếu `--extra cu128`
thì `uv` có thể cài lại PyTorch bản không có GPU.

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
| `--agent.run-name ...` | Tên thư mục kết quả, để dễ tìm. Nên ghi cả phương pháp và seed |

**Đang chạy thì trông như thế nào?** Terminal in ra liên tục các khối `Learning iteration 15/2000`
kèm các con số. Dòng `ETA` cho biết còn bao lâu. Với DAgger, cứ 5 vòng (trong 110 vòng đầu) có một
vòng chậm hơn khoảng 25 giây: đó là lúc "thầy" MPC gắn nhãn.

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

### 5.3. Train nhiều seed (bắt buộc khi so sánh)

Cùng một phương pháp, mỗi lần train ra một kết quả hơi khác: có lần robot "tìm ra" dáng đi nhanh
sớm hơn, có lần muộn hơn vài trăm vòng. **Một lần train không đủ để kết luận phương pháp nào hơn.**
Mỗi phương pháp nên chạy **ít nhất 2, tốt nhất 3 seed**. Lệnh sau chạy lần lượt seed 1, 2, 3:

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
    └── params/                                       ← cấu hình đã dùng (agent.yaml, env.yaml)
```

`model_1999.pt` là kết quả cuối cùng. Xem tên các lần chạy:

```bash
ls logs/rsl_rl/g1_velocity_2k/
```

### 6.2. Xem biểu đồ học

```bash
uv run --extra cu128 tensorboard --logdir logs/rsl_rl/g1_velocity_2k
```

Mở trình duyệt vào địa chỉ **http://localhost:6006**. Các biểu đồ đáng xem:

| Biểu đồ | Ý nghĩa | Tốt khi |
|---|---|---|
| `Episode_Reward/track_linear_velocity` | Robot bám vận tốc tốt đến đâu (tối đa 2,0) | Lên khoảng 1,5 |
| `Train/mean_episode_length` | Robot đứng được bao lâu trước khi ngã (tối đa 1000) | Gần 1000 |
| `Train/mean_reward` | Tổng điểm thưởng | Tăng dần |
| `Loss/mpc/bc_loss` (chỉ DAgger) | Robot còn khác "thầy" bao nhiêu | Về 0 sau vòng 150 (lúc ngừng bắt chước) |

Xem xong thì quay lại terminal, nhấn `Ctrl + C` để tắt TensorBoard.

### 6.3. Đo vận tốc thực tế của robot

Lệnh này cho robot đã học chạy thẳng ở các lệnh 0,5 / 1,0 / 1,5 m/s với **64 robot**: 1 giây đầu để
tăng tốc, rồi **đo trong 10 giây**. Thay `TÊN_THƯ_MỤC` bằng tên lần chạy của bạn:

```bash
uv run --extra cu128 mpopi-eval --task Mpopi-G1-2k-PPO --controllers policy --num-envs 64 --steps 550 --settle-steps 50 --checkpoint logs/rsl_rl/g1_velocity_2k/TÊN_THƯ_MỤC/model_1999.pt
```

Dùng `--task Mpopi-G1-2k-PPO` cho **mọi** phương pháp: robot và mạng giống nhau, và cách này không
phải dựng "thầy" MPC khi đánh giá. Thêm `--out ket_qua.json` nếu muốn lưu kết quả ra file.

Cách đọc kết quả ở dòng lệnh 1,5 m/s:

| Cột | Ý nghĩa | Tốt khi |
|---|---|---|
| `speed` | Vận tốc thật đạt được (m/s) | Gần 1,5 (từ 1,45 trở lên) |
| `\|err\|` | Sai lệch trung bình so với lệnh (m/s) | Nhỏ, khoảng 0,04–0,06 |
| `falls/env` | Số lần ngã mỗi robot | 0 |

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
| "Thầy" MPC gắn nhãn lâu hơn (tới vòng 300) | `--agent.algorithm.mpopi.mpc.collect-iterations 300 --agent.algorithm.mpopi.mpc.bc-iterations 340` |
| Ít env phụ cho "thầy" MPC hơn | `--agent.algorithm.mpopi.mpc.num-envs 32` |
| Lưu lên Weights & Biases thay vì máy | Bỏ `--agent.logger tensorboard` (cần đăng nhập W&B trước) |
| Xem mọi tùy chọn | `uv run --extra cu128 mpopi-train Mpopi-G1-2k-PPO --help` |

**Lưu ý:** đổi cấu hình thì kết quả **không còn so sánh trực tiếp được** với các lần chạy dùng cấu
hình chuẩn. Cấu hình chuẩn của từng phương pháp nằm trong `src/mpopi_train/presets.py`; cấu hình
thực sự đã dùng của mỗi lần chạy được lưu trong `params/agent.yaml` của thư mục lần chạy đó.

---

## 8. Cách B: chạy trên Kaggle (không cần GPU riêng)

Kaggle cho dùng miễn phí 2 GPU T4, khoảng 30 giờ mỗi tuần.

### 8.1. Các notebook

| File (trong thư mục `notebooks/`) | Dùng để |
|---|---|
| `mpc_g1_train_kaggle.ipynb` | Train và đánh giá các phương pháp (tối đa 12 giờ một phiên) |
| `mpc_g1_eval_kaggle.ipynb` | Chỉ đánh giá lại checkpoint của các lần train trước |

Lấy file từ thư mục code đã clone, hoặc trên GitHub: mở file, nhấn nút *Download raw file*.

### 8.2. Train trên Kaggle

1. Tạo tài khoản ở **https://www.kaggle.com** và **xác minh số điện thoại** (*Settings → Phone
   verification*). Không xác minh thì không bật được Internet và GPU.
2. Trên Kaggle: **Create → New Notebook**, rồi **File → Import Notebook** và chọn
   `mpc_g1_train_kaggle.ipynb`.
3. Ở panel bên phải, mục **Settings**:
   - **Accelerator: GPU T4 x2.** Không chọn P100, vì không chạy được.
   - **Internet: On.**
4. Mở ô **"0. Cấu hình"** nếu muốn chọn phương pháp (`TRAIN`) hoặc seed (`SEEDS`). Mặc định
   notebook train cả 4 phương pháp với seed 1 (khoảng 4 giờ trên 2 GPU).
5. Nhấn **Save Version** (góc trên bên phải) → chọn **Save & Run All (Commit)** → **Save**.
   Notebook sẽ chạy nền, tối đa 12 giờ. Có thể tắt trình duyệt.
6. Khi chạy xong (trạng thái chuyển thành *Complete*), mở phiên bản đó → tab **Output** → tải
   **`mpc_g1_train_results.zip`**. Trong đó có biểu đồ học (`train_curves.png`), biểu đồ thầy–trò
   (`teacher_gap.png`), bảng vận tốc `ket_qua.txt` và video.

Nếu 12 giờ không đủ, notebook tự bỏ qua phần còn thiếu. Để chạy tiếp: tạo phiên bản mới, thêm output
của phiên bản cũ làm *Input* (**Add Input → Your Work**), rồi chạy lại. Notebook chỉ train phần còn
thiếu.

### 8.3. Quy tắc quan trọng: mỗi thí nghiệm một notebook riêng

Mỗi lần chạy một thí nghiệm **khác** (đổi phương pháp, đổi cấu hình, đổi code), hãy **Import
Notebook thành một notebook mới** trên Kaggle, đặt tên rõ ràng (ví dụ `g1-train-4-methods`,
`g1-bc-300`). **Đừng lưu đè thành phiên bản mới của notebook cũ.**

Lý do: khi một notebook khác cần dùng kết quả (ví dụ notebook đánh giá lại ở mục 8.4), Kaggle chỉ
gắn output của **phiên bản mới nhất**. Lưu nhiều thí nghiệm chồng lên một notebook thì kết quả của
các thí nghiệm cũ coi như không lấy lại được.

### 8.4. Đánh giá lại checkpoint cũ

Checkpoint cuối (`model_1999.pt`) của mỗi lần train vẫn nằm trong Output của notebook train (không
có trong file zip). Để đánh giá lại:

1. Import `mpc_g1_eval_kaggle.ipynb` thành notebook mới, bật **GPU T4 x2** và **Internet On**.
2. **Add Input → Your Work → Notebooks**, thêm các notebook train cần đánh giá.
3. **Save Version → Save & Run All**. Ô "3. Tìm checkpoint" in ra danh sách lần train tìm được;
   kiểm tra xem đã đủ chưa.
4. Tải **`mpc_g1_eval_results.zip`** ở tab Output.

---

## 9. Gặp lỗi thì làm gì

| Thông báo / hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `nvidia-smi: command not found` | Không có card NVIDIA hoặc chưa cài driver | Dùng Kaggle (mục 8) |
| `GPU: False` ở bước 4 | Thiếu `--extra cu128`, hoặc driver quá cũ | Chạy lại `uv sync --extra cu128`; cập nhật driver NVIDIA |
| `CUDA out of memory` | GPU không đủ bộ nhớ | Thêm `--env.scene.num-envs 2048` (hoặc 1024) |
| `invalid choice: 'Mpopi-G1-2k-...'` hoặc `mpopi-train: command not found` | Sai nhánh hoặc sai thư mục | `cd ~/mjlab_MPOPI` rồi `git checkout mpc-stage1` |
| Hỏi đăng nhập `wandb` | Quên `--agent.logger tensorboard` | Thêm tùy chọn đó vào lệnh |
| `uv` tải lại PyTorch mỗi lần chạy | Có lệnh thiếu `--extra cu128` | Luôn dùng `uv run --extra cu128 ...` |
| Kết quả hai lần train cùng cấu hình khác nhau | Bình thường trong học tăng cường | Chạy nhiều seed và so khoảng giá trị (mục 5.3) |
| Kaggle: không bật được GPU hoặc Internet | Chưa xác minh số điện thoại | Xác minh trong *Settings* của tài khoản Kaggle |
| Kaggle: lỗi CUDA trên P100 | P100 không được hỗ trợ | Chọn **GPU T4 x2** |
| Kaggle: notebook đánh giá không thấy checkpoint | Input là phiên bản mới nhất, không phải phiên bản đã train | Xem mục 8.3 |
| Terminal đứng yên rất lâu ở lần chạy đầu | Đang tải thư viện hoặc biên dịch | Chờ 5–10 phút; chỉ xảy ra lần đầu |

Vẫn không được: chụp **toàn bộ** thông báo lỗi trong terminal (cuộn lên đầu lỗi) và gửi cho người
phụ trách dự án.

---

## 10. Tóm tắt nhanh (khi đã cài xong)

```bash
cd ~/mjlab_MPOPI
uv run --extra cu128 mpopi-train Mpopi-G1-2k-Replay-IS-DAgger --agent.logger tensorboard --agent.seed 1 --agent.run-name Replay-IS-DAgger_s1
uv run --extra cu128 tensorboard --logdir logs/rsl_rl/g1_velocity_2k
uv run --extra cu128 mpopi-eval --task Mpopi-G1-2k-PPO --controllers policy --num-envs 64 --steps 550 --settle-steps 50 --checkpoint logs/rsl_rl/g1_velocity_2k/TÊN_THƯ_MỤC/model_1999.pt
```
