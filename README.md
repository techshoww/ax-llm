# AX-LLM

![GitHub License](https://img.shields.io/github/license/AXERA-TECH/ax-llm)

| Platform | Build Status |
| -------- | ------------ |
| AX650    | ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/AXERA-TECH/ax-llm/build_650.yml?internvl2)|

## 简介

**AX-LLM** 由 **[爱芯元智](https://www.axera-tech.com/)** 主导开发。该项目用于探索业界常用 **LLM(Large Language Model)** 在已有芯片平台上落地的可行性和相关能力边界，**方便**社区开发者进行**快速评估**和**二次开发**自己的 **LLM 应用**。

### 已支持芯片

- AX650A/AX650N
  - SDK ≥ v1.45.0_P31

### 已支持模型

- Qwen2.5-VL-3B-Instruct

### 获取地址

comming soon

## 源码编译

-  clone 本项目  
    ```shell
    git clone  https://github.com/AXERA-TECH/ax-llm.git
    cd ax-llm
    ```
- clone `ax650n_bsp_sdk` 代码  
    ```shell
    git cloen https://github.com/AXERA-TECH/ax650n_bsp_sdk
    ```
- 仔细阅读 `build.sh` ，并在 `build.sh` 正确修改 `BSP_MSP_DIR` 变量后(该变量表示`ax650n_bsp_sdk`代码位置)，运行编译脚本  
    ```shell
    ./build.sh
    ```
- 正确编译后，`build/install/bin` 目录，应有以下文件（百度网盘中有预编译的可执行程序）
  ```
  $ tree install/bin/
    install/bin/
    ├── main
    ├── run_bf16.sh
    └── run_qwen_1.8B.sh
  ```
  
## 运行示例

### 1. 图像理解

![demo.jpg](assets/demo.jpg)

#### 1. 首先启动 HTTP Tokenizer Server  
```
cd scripts
python qwen2_tokenizer_image_448.py --host {your host} --port {your port}   # 和 run_qwen2_5vl_image.sh 中一致
```

#### 2. 在板子上运行模型  
1) 先修改 `run_qwen2_5vl_image.sh` 中的http host.  
2) 将 `scripts/run_qwen2_5vl_image.sh`, `src/post_config.json` ,`build/install/bin/main`, `assets/demo.jpg` 拷贝到爱芯板子上  
3) 运行 `run_qwen2_5vl_image.sh`  
```shell

```

### 2. 视频理解

<div style="
    display: grid;
    grid-template-columns: repeat(4, 1fr);  /* 4列等宽 */
    grid-template-rows: repeat(2, 1fr);     /* 2行等高 */
    gap: 10px;                              /* 图片间距 */
    width: 80%;                             /* 容器宽度 */
    margin: 0 auto;                         /* 居中显示 */
">
    <img src="demo_cv308/frame_0075.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0077.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0079.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0081.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0083.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0085.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0087.jpg" style="width: 100%; height: 100%; object-fit: cover;">
    <img src="demo_cv308/frame_0089.jpg" style="width: 100%; height: 100%; object-fit: cover;">
</div>

#### 1. 首先启动 HTTP Tokenizer Server  
```
cd scripts
python qwen2_tokenizer_video_308.py --host {your host} --port {your port}   # 和 run_qwen2_5vl_video.sh 中一致
```

#### 2. 在板子上运行模型  
1) 先修改 `run_qwen2_5vl_video.sh` 中的http host.  
2) 将 `scripts/run_qwen2_5vl_video.sh`, `src/post_config.json` ,`build/install/bin/main`, `demo_cv308` 拷贝到爱芯板子上  
3) 运行 `run_qwen2_5vl_video.sh`  
```shell

```

## 图像理解推理速度  
| Stage | Time |
|------|------|
| Image Encoder (448x448) | 790 ms  | 
| Prefill (320) |  43535.27 ms    |
| Decode  |   token/s |

## 视频理解推理速度  
| Stage | Time |
|------|------|
| Image Encoder (8x308x308) |  ms  | 
| Prefill (512) |   ms    |
| Decode  |   token/s |

## Reference

- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## 技术讨论

- Github issues
- QQ 群: 139953715
