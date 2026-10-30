# ReID using OpenVINO

简体中文 | [English](README_en.md)

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-Inference%20Engine-0071C5?logo=intel&logoColor=white)](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/overview.html)
[![Stars](https://img.shields.io/github/stars/yiyiliu0415/ReID_using-openvino?style=flat&color=yellow)](https://github.com/yiyiliu0415/ReID_using-openvino/stargazers)

**给定一张目标人物的照片，从长时间监控视频中检索出他/她出现过的所有片段。**

项目把行人检测、多目标跟踪（MOT）与行人重识别（ReID）串成一条流水线，推理侧全部跑在
Intel OpenVINO 上，因此**没有独立显卡的普通办公 PC 也能跑**。输出是一份带时间戳的文本报告，
直接告诉你目标在第几路摄像头、第几秒出现过。

![pipeline preview](common/show_pic/image.png)

---

## 目录

- [为什么用它](#为什么用它)
- [工作原理](#工作原理)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [参数说明](#参数说明)
- [输出结果](#输出结果)
- [目录结构](#目录结构)
- [常见问题](#常见问题)
- [致谢](#致谢)

---

## 为什么用它

| | |
|---|---|
| **CPU 就能跑** | 检测与 ReID 特征提取都通过 OpenVINO IR 模型推理，Intel CPU 上即可达到可用帧率 |
| **先跟踪再识别** | 先用跟踪器把同一个人串成轨迹，再对轨迹做特征匹配，避免逐帧重复比对 |
| **可跳帧加速** | `--skip N` 每 N 帧处理一次，长视频处理时间线性下降 |
| **结果可复现** | 中间结果落盘为 `detections.npy` / `rm.npy`，改判定逻辑时无需重新推理 |
| **多路视频** | `-v` 支持一次传入多个视频，输出统一时间轴报告 |

## 工作原理

```
视频输入 ──► 行人检测(OpenVINO Detector) ──► 多目标跟踪(MCT / SCT)
                                                    │
                                           每帧 bbox ──► ReID 特征提取(hr18)
                                                    │
目标照片 ──► 特征库 ────────────────────────────► 余弦相似度匹配
                                                    │
                                            top-k 命中 ──► 时间戳文本报告
```

- **检测**：`utils/network_wrappers.py` 中的 `Detector` 封装 OpenVINO Inference Engine，默认使用
  `model/detection.xml`；`common/python/models/` 下另外保留了 SSD、RetinaFace、YOLO、CenterNet
  等模型的封装，方便替换检测器。
- **跟踪**：`mc_tracker/mct.py`（多摄像头跟踪，跨镜头关联）与 `mc_tracker/sct.py`（单摄像头跟踪），
  由 `tracker.py` 中的 `Tracker` 类调度。
- **ReID**：默认特征提取模型为 `model/hr18.xml`（ResNet18 骨干），对每条轨迹的代表帧提取特征后
  与目标照片特征比对。
- **判定**：`--top_k` 控制取相似度最高的前 k 条轨迹，`path_use.py` 负责把帧号换算成
  `时:分:秒` 并写出报告。

## 环境准备

- Python 3.7+
- Intel OpenVINO（Windows / Linux 安装方式不同，见[官网](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/overview.html)）
- 依赖：

```bash
pip install -r requirements.txt
```

> `requirements.txt` 里对 `numpy<1.20`、`scipy~=1.5.4` 有版本约束，建议新建虚拟环境后再安装，
> 避免和系统里已有的科学计算环境冲突。

### 模型与示例数据

仓库不包含模型权重。模型文件、示例视频与示例目标图已打包，可从
[百度网盘](https://pan.baidu.com/s/1nkM6b2jDnjJHJTUZV-qGDw?pwd=oj8k)（提取码 `oj8k`）下载，
解压后直接覆盖到项目根目录即可，目录结构如下：

```
model/
├── detection.xml / detection.bin   # 行人检测模型
└── hr18.xml / hr18.bin             # ReID 特征提取模型
```

## 快速开始

### 1. 新建 `configs/configs.json`

这个文件记录**每路视频对应的地点和开拍时间**，用于把视频内的相对帧号还原成真实时刻。
例如超市门口的监控 `0.mp4` 从 8:00 开始拍摄：

```json
{
    "place": {
        "supermarket": {
            "0": 28800
        }
    }
}
```

`"0"` 是视频文件名（不含扩展名），`28800` 是开拍时刻的秒数（8 × 60 × 60）。

### 2. 放入视频与目标照片

```bash
mkdir -p video_input target_input
# 视频：video_input/0.mp4
# 目标照片：target_input/xxx.jpg（单人、清晰正脸或全身照效果最好）
```

### 3. 运行

```bash
python ReID_main.py -v ./video_input -t ./target_input \
    --data_output ./data_output --top_k 50 --skip 10
```

### 4. 复用已有推理结果

如果只是想调整判定阈值、不想重新跑推理，可以复用上次的 `.npy`：

```bash
python ReID_main.py -v ./video_input -t ./target_input \
    --is_only_txt True --npy_path ./data_output/2 \
    --top_k 50 --skip 10
```

## 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-v, --video_input` | 必填 | 输入视频路径，可传多个（`nargs='+'`） |
| `-t, --target_input` | 必填 | 目标人物照片路径，可传多个 |
| `--data_output` | `./data_output/2` | 输出目录，存放报告与中间结果 |
| `--npy_path` | `./data_output/2` | 已有 `.npy` 结果的路径，配合 `--is_only_txt` 使用 |
| `--json_path` | `./configs/configs.json` | 视频地点/起始时间配置 |
| `-m, --m_detector` | `./model/detection.xml` | 检测模型（OpenVINO IR） |
| `--m_reid_xml` / `--m_reid_bin` | `./model/hr18.xml` / `.bin` | ReID 特征提取模型 |
| `--config` | `./configs/person.py` | 场景配置；做车辆检索时换成 `configs/vehicle.py` |
| `--t_detector` | `0.95` | 检测置信度阈值，漏检多就调低 |
| `--top_k` | `-1` | 取相似度最高的前 k 条轨迹；**过大或过小都会影响结果**，建议 30–50 |
| `--skip` | `10` | 每 N 帧处理一次；不跳帧填 `1` |
| `-d, --device` | `CPU` | 推理设备（`CPU` / `GPU` / `MYRIAD`） |
| `-l, --cpu_extension` | 空 | 自定义层 CPU 扩展库（旧版 OpenVINO 才需要） |
| `-u, --utilization_monitors` | 空 | 绑定的硬件监控设备，如 `GPU,0` |
| `--is_save` | `True` | 是否保存可视化视频 |
| `--is_tracker` | `True` | 是否启用跟踪（关掉则逐帧比对，慢很多） |
| `--is_txt` | `True` | 是否输出文本报告 |
| `--is_only_txt` | `False` | 跳过推理，仅根据已有 `.npy` 生成报告 |
| `--loop` | `False` | 循环播放输入视频 |

## 输出结果

```
data_output/
├── message/            # 文本报告：摄像头 / 时间段 / 相似度
├── *.npy               # detections.npy、rm.npy 等中间结果
└── *.mp4               # 带 bbox 与 ID 的可视化视频（--is_save True 时）
```

报告形如 `[supermarket][0] 08:13:24 - 08:14:02  score: 0.87`，表示该目标在超市门口 0 号视频的
08:13:24 至 08:14:02 出现。

## 目录结构

```
ReID_using-openvino/
├── ReID_main.py           # 主入口：参数解析 + 流水线调度
├── tracker.py             # Tracker：跟踪 + ReID 特征匹配
├── path_use.py            # 时间换算、报告写出
├── visualize.py           # 结果可视化
├── configs/
│   ├── person.py          # 行人场景配置（跟踪器超参、可视化）
│   └── vehicle.py         # 车辆场景配置
├── mc_tracker/
│   ├── mct.py             # 多摄像头跟踪
│   └── sct.py             # 单摄像头跟踪
├── utils/
│   ├── network_wrappers.py  # OpenVINO 检测器封装
│   ├── video.py             # 多路视频读取、CLAHE 归一化
│   ├── visualization.py     # 画框、堆叠画面
│   └── ...
└── common/python/         # OpenVINO demo 公共组件与模型封装
```

## 常见问题

**`configs/configs.json` 找不到？**
仓库里没有这个文件，需要按上文第 1 步自行创建。不配置也能跑，但报告里的时间会是视频内相对时间。

**结果里同一个人被拆成多条轨迹？**
跟踪器参数在 `configs/person.py`：`mct_config.global_match_thresh`（跨镜头关联阈值）、
`sct_config.match_threshold`（同一镜头内关联阈值）。适当调低更容易把轨迹连起来。

**目标完全检索不到？**
依次检查：`--t_detector` 是否过高导致漏检；`--top_k` 是否过小；目标照片是否与监控画面
视角/光照差异过大；`--skip` 是否跳过了目标出现的短片段。

**内存占用高？**
长视频建议先切段，或调大 `--skip`；`common/python` 下的模型封装按需加载，不会全部驻留。

## 致谢

检测与跟踪框架参考 Intel OpenVINO 官方 multi-camera-multi-person-tracking 示例，在其基础上接入了
ReID 目标检索与时间戳报告。

欢迎提 issue 交流；如果这个项目帮你省了时间，点个 star 就好。
