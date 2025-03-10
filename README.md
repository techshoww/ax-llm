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

- InternVL2-1B
- SmolVLM-256M-Instruct

### 获取地址

- InternVL2-1B [百度网盘](https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e)
- SmolVLM-256M-Instruct [下载地址](https://github.com/techshoww/ax-llm/releases/download/v1.0.0/SmolVLM-256M-Instruct-AX650.tar.gz) 。推荐这里面的模型[AXERA huggingface](https://huggingface.co/AXERA-TECH/SmolVLM-256M-Instruct)，这里面的模型编译的prefill_len更小，跑的更快。

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

### SmolVLM-256M-Instruct

![demo.jpg](assets/demo.jpg)

#### 1. 首先启动 HTTP Tokenizer Server  
```
cd scripts
python smolvlm_tokenizer_512.py  --host {your host} --port {your port}   # 和 run_smolvlm.sh 中一致
```

#### 2. 在板子上运行模型  
1) 先修改 `run_smolvlm.sh` 中的http host.  
2) 将 `scripts/run_smolvlm.sh`, `src/post_config.json` ,`build/install/bin/main`, `assets/demo.jpg` 拷贝到爱芯板子上  
3) 运行 `run_smolvlm.sh`  
```shell
root@ax650 ~/SmolVLM-256M-Instruct-Infer # bash run_smolvlm.sh 
[I][                            Init][ 106]: LLM init start
bos_id: 1, eos_id: 49279
  2% | █                                 |   1 /  34 [0.01s<0.27s, 125.00 count/s] tokenizer init ok[I][                            Init][  26]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  34 /  34 [1.59s<1.59s, 21.40 count/s] init vpm axmodel ok,remain_cmm(3498 MB)B)
[I][                            Init][ 254]: max_token_len : 1023
[I][                            Init][ 259]: kv_cache_size : 192, kv_cache_num: 1023
[I][                            Init][ 267]: prefill_token_num : 128
[I][                            Init][ 269]: vpm_height : 512,vpm_width : 512
[I][                            Init][ 278]: LLM init ok
Type "q" to exit, Ctrl+c to stop current running
prompt >> Can you describe this image?
image >> assets/demo.jpg
[I][                          Encode][ 337]: image encode time : 119.578003 ms, size : 36864
[I][                             Run][ 548]: ttft: 57.75 ms
 The image depicts a large, historic statue of Liberty, located in New York City. The statue is a prominent landmark and is known for its iconic presence in the city. The statue is located on a pedestal that is surrounded by a large, circular base. The base of the statue is made of stone and is painted in a light blue color. The statue is surrounded by a large, circular ring that encircles the base.

The statue is made of bronze and is quite large, measuring approximately 100 feet in height. The statue is mounted on a pedestal that is made of stone and is painted in a light blue color. The pedestal is rectangular and is supported by a series of columns. The columns are made of stone and are painted in a light blue color. The statue is surrounded by a large, circular ring that encircles the base.

In the background, there is a large cityscape with a variety of buildings and structures. The sky is clear and blue, indicating that it is a sunny day. The buildings are tall and have a modern architectural style, with large windows and balconies. The buildings are mostly made of glass and steel, and they are painted in a variety of colors.

There are a few trees and bushes visible in the foreground, which are located on the left side of the image. The trees are green and appear to be healthy. There is also a small, white building visible in the background, which is likely a hotel or a small office.

The overall atmosphere of the image is one of peace and tranquility. The statue is a symbol of freedom and liberty, and the surrounding buildings and structures add to the sense of the city's historical and cultural significance.

In summary, the image depicts the Statue of Liberty, a large, historic statue located in New York City. The statue is a prominent landmark and is surrounded by a large, circular ring that encircles the base. The statue is painted in a light blue color and is mounted on a pedestal that is surrounded by a large, circular ring. The statue is surrounded by a large, circular ring that encircles the base. The background includes a large cityscape with tall buildings and a clear blue sky. The overall atmosphere of the image is one of peace and tranquility.

[N][                             Run][ 687]: hit eos,avg 76.88 token/s
```

## 推理速度  
| Stage | Time |
|------|------|
| Image Encoder (512x512) | 120 ms  | 
| Prefill |  57ms    |
| Decode  |  77 token/s |

## Reference

- [InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B)
- [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
## 技术讨论

- Github issues
- QQ 群: 139953715
