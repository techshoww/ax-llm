#include "ax_model_runner_ax650.hpp"
#include "string.h"
#include "fstream"
#include "memory"
// #include "utilities/file.hpp"
// #include <ax_sys_api.h>
// #include <ax_ivps_api.h>
// #include <ax_engine_api.h>
#include <fcntl.h>
#include "memory_utils.hpp"
#include "sample_log.h"

// #include <axcl/native/ax_sys_api.h>
// #include <axcl.h>
#include "utils/axcl_manager.h"

#define AX_CMM_ALIGN_SIZE 128

static const char *AX_CMM_SESSION_NAME = "npu";

typedef enum
{
    AX_ENGINE_ABST_DEFAULT = 0,
    AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> INPUT_OUTPUT_ALLOC_STRATEGY;

static void print_io_info(std::vector<ax_runner_tensor_t> &input, std::vector<ax_runner_tensor_t> &output)
{
    printf("\ninput size: %ld\n", input.size());
    for (uint32_t i = 0; i < input.size(); ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = input[i];
        printf("    name: \e[1;32m%8s", info.sName.c_str());

        std::string dt = "unknown";

        printf(" \e[1;31m[%s] ", dt.c_str());

        std::string ct = "unknown";

        printf("\e[1;31m[%s]", ct.c_str());

        printf(" \n        \e[1;31m");

        for (int s = 0; s < info.vShape.size(); s++)
        {
            printf("%d", info.vShape[s]);
            if (s != info.vShape.size() - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n\n");
    }

    printf("\noutput size: %ld\n", output.size());
    for (uint32_t i = 0; i < output.size(); ++i)
    {
        // print shape info,like [batchsize x channel x height x width]
        auto &info = output[i];
        printf("    name: \e[1;32m%8s \e[0m\n        \e[1;31m", info.sName.c_str());
        for (int s = 0; s < info.vShape.size(); s++)
        {
            printf("%d", info.vShape[s]);
            if (s != info.vShape.size() - 1)
            {
                printf(" x ");
            }
        }
        printf("\e[0m\n\n");
    }
}

typedef struct
{
    int nIndex;
    int nSize;
    void *pBuf;

    std::string Name;

    axclrtEngineIODims dims;
} AXCL_IO_BUF_T;

typedef struct
{
    uint32_t nInputSize;
    uint32_t nOutputSize;
    AXCL_IO_BUF_T *pInputs;
    AXCL_IO_BUF_T *pOutputs;
} AXCL_IO_DATA_T;

static void free_io_index(AXCL_IO_BUF_T *pBuf, size_t index, int _devid)
{
    for (size_t i = 0; i < index; ++i)
    {
        axcl_Free(pBuf[i].pBuf, _devid);
    }
}

static void free_io(AXCL_IO_DATA_T *io_data, int _devid)
{
    for (size_t j = 0; j < io_data->nInputSize; ++j)
    {
        axcl_Free(io_data->pInputs[j].pBuf, _devid);
    }
    for (size_t j = 0; j < io_data->nOutputSize; ++j)
    {
        axcl_Free(io_data->pOutputs[j].pBuf, _devid);
    }
    delete[] io_data->pInputs;
    delete[] io_data->pOutputs;
}

static inline int prepare_io(uint64_t handle, uint64_t context, axclrtEngineIOInfo io_info, axclrtEngineIO io, AXCL_IO_DATA_T *io_data, INPUT_OUTPUT_ALLOC_STRATEGY strategy, int _devid)
{
    memset(io_data, 0, sizeof(AXCL_IO_DATA_T));

    auto inputNum = axcl_EngineGetNumInputs(io_info, _devid);
    auto outputNum = axcl_EngineGetNumOutputs(io_info, _devid);
    io_data->nInputSize = inputNum;
    io_data->nOutputSize = outputNum;
    io_data->pInputs = new AXCL_IO_BUF_T[inputNum];
    io_data->pOutputs = new AXCL_IO_BUF_T[outputNum];

    // 1. alloc inputs
    for (int32_t i = 0; i < inputNum; i++)
    {
        auto bufSize = axcl_EngineGetInputSizeByIndex(io_info, 0, i, _devid);
        void *devPtr = nullptr;
        axclError ret = 0;
        if (AX_ENGINE_ABST_DEFAULT == strategy.first)
        {
            ret = axcl_Malloc(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }
        else
        {
            ret = axcl_MallocCached(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }

        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i, _devid);
            fprintf(stderr, "Malloc input(index: %d, size: %ld) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
        std::vector<char> tmp(bufSize, 0);
        axcl_Memcpy(devPtr, tmp.data(), bufSize, axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _devid);
        // axclrtMemset(devPtr, 0, bufSize);

        axclrtEngineIODims dims;
        ret = axcl_EngineGetInputDims(io_info, 0, i, &dims, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i, _devid);
            fprintf(stderr, "Get input dims(index: %d) failed! ret=0x%x\n", i, ret);
            return -1;
        }

        io_data->pInputs[i].nIndex = i;
        io_data->pInputs[i].nSize = bufSize;
        io_data->pInputs[i].pBuf = devPtr;
        io_data->pInputs[i].dims = dims;
        io_data->pInputs[i].Name = axcl_EngineGetInputNameByIndex(io_info, i, _devid);
        ret = axcl_EngineSetInputBufferByIndex(io, i, devPtr, bufSize, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i, _devid);
            fprintf(stderr, "Set input buffer(index: %d, size: %lu) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
    }

    // 2. alloc outputs
    for (int32_t i = 0; i < outputNum; i++)
    {
        auto bufSize = axcl_EngineGetOutputSizeByIndex(io_info, 0, i, _devid);
        void *devPtr = NULL;
        axclError ret = 0;
        if (AX_ENGINE_ABST_DEFAULT == strategy.first)
        {
            ret = axcl_Malloc(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }
        else
        {
            ret = axcl_MallocCached(&devPtr, bufSize, axclrtMemMallocPolicy::AXCL_MEM_MALLOC_HUGE_FIRST, _devid);
        }

        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i, _devid);
            fprintf(stderr, "Malloc output(index: %d, size: %ld) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
        std::vector<char> tmp(bufSize, 0);
        axcl_Memcpy(devPtr, tmp.data(), bufSize, axclrtMemcpyKind::AXCL_MEMCPY_HOST_TO_DEVICE, _devid);
        // axclrtMemset(devPtr, 0, bufSize);
        axclrtEngineIODims dims;
        ret = axcl_EngineGetOutputDims(io_info, 0, i, &dims, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i, _devid);
            fprintf(stderr, "Get output dims(index: %d) failed! ret=0x%x\n", i, ret);
            return -1;
        }

        io_data->pOutputs[i].nIndex = i;
        io_data->pOutputs[i].nSize = bufSize;
        io_data->pOutputs[i].pBuf = devPtr;
        io_data->pOutputs[i].dims = dims;
        io_data->pOutputs[i].Name = axcl_EngineGetOutputNameByIndex(io_info, i, _devid);
        ret = axcl_EngineSetOutputBufferByIndex(io, i, devPtr, bufSize, _devid);
        if (ret != 0)
        {
            free_io_index(io_data->pOutputs, i, _devid);
            fprintf(stderr, "Set output buffer(index: %d, size: %lu) failed! ret=0x%x\n", i, bufSize, ret);
            return -1;
        }
    }

    return 0;
}

struct ax_joint_runner_ax650_handle_t
{
    uint64_t handle = 0;
    uint64_t context = 0;
    axclrtEngineIOInfo io_info = 0;
    axclrtEngineIO io = 0;
    AXCL_IO_DATA_T io_data = {0};

    // int algo_width, algo_height;
    // int algo_colorformat;
};

int ax_runner_ax650::sub_init()
{
    // 4. create context
    int ret = axcl_EngineCreateContext(m_handle->handle, &m_handle->context, _devid);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateContext");
        return ret;
    }
    // fprintf(stdout, "Engine creating context is done.\n");

    // 5. set io

    ret = axcl_EngineGetIOInfo(m_handle->handle, &m_handle->io_info, _devid);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_GetIOInfo");
        return ret;
    }
    // fprintf(stdout, "Engine get io info is done. \n");

    // 4. create io
    ret = axcl_EngineCreateIO(m_handle->io_info, &m_handle->io, _devid);
    if (ret != 0)
    {
        axcl_EngineUnload(m_handle->handle, _devid);
        fprintf(stderr, "Create io failed. ret=0x%x\n", ret);
        return -1;
    }

    // fprintf(stdout, "Engine creating io is done. \n");

    // 6. alloc io
    if (!_parepare_io)
    {
        auto malloc_strategy = std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_DEFAULT);
        ret = prepare_io(m_handle->handle, m_handle->context, m_handle->io_info, m_handle->io, &m_handle->io_data, malloc_strategy, _devid);
        if (ret != 0)
        {
            free_io(&m_handle->io_data, _devid);
            axcl_EngineDestroyIO(m_handle->io, _devid);
            axcl_EngineUnload(m_handle->handle, _devid);

            fprintf(stderr, "prepare_io failed.\n");
            return ret;
        }

        for (size_t i = 0; i < m_handle->io_data.nOutputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = m_handle->io_data.pOutputs[i].Name;
            tensor.nSize = m_handle->io_data.pOutputs[i].nSize;
            for (size_t j = 0; j < m_handle->io_data.pOutputs[i].dims.dimCount; j++)
            {
                tensor.vShape.push_back(m_handle->io_data.pOutputs[i].dims.dims[j]);
            }
            tensor.phyAddr = (unsigned long long)m_handle->io_data.pOutputs[i].pBuf;
            tensor.pVirAddr = malloc(tensor.nSize);
            memset(tensor.pVirAddr, 0, tensor.nSize);
            moutput_tensors.push_back(tensor);
        }

        for (size_t i = 0; i < m_handle->io_data.nInputSize; i++)
        {
            ax_runner_tensor_t tensor;
            tensor.nIdx = i;
            tensor.sName = m_handle->io_data.pInputs[i].Name;
            tensor.nSize = m_handle->io_data.pInputs[i].nSize;
            for (size_t j = 0; j < m_handle->io_data.pInputs[i].dims.dimCount; j++)
            {
                tensor.vShape.push_back(m_handle->io_data.pInputs[i].dims.dims[j]);
            }
            tensor.phyAddr = (unsigned long long)m_handle->io_data.pInputs[i].pBuf;
            // tensor.pVirAddr = m_handle->io_data.pInputs[i].pVirAddr;
            tensor.pVirAddr = malloc(tensor.nSize);
            memset(tensor.pVirAddr, 0, tensor.nSize);
            minput_tensors.push_back(tensor);
        }
        _parepare_io = true;
    }
    else
    {
    }
    // print_io_info(minput_tensors, mtensors);

    return ret;
}

int ax_runner_ax650::init(const char *model_file, int devid, bool use_mmap)
{
    if (use_mmap)
    {
        MMap model_buffer(model_file);
        if (!model_buffer.data())
        {
            ALOGE("mmap");
            return -1;
        }
        auto ret = init((char *)model_buffer.data(), model_buffer.size(), devid);
        model_buffer.close_file();
        return ret;
    }
    else
    {
        char *model_buffer;
        size_t len;
        if (!read_file(model_file, &model_buffer, &len))
        {
            ALOGE("read_file");
            return -1;
        }
        auto ret = init(model_buffer, len, devid);
        delete[] model_buffer;
        return ret;
    }
}

int ax_runner_ax650::init(char *model_buffer, size_t model_size, int devid)
{
    if (!m_handle)
    {
        m_handle = new ax_joint_runner_ax650_handle_t;
    }
    memset(m_handle, 0, sizeof(ax_joint_runner_ax650_handle_t));
    _devid = devid;
    // 3. create handle
    void *devMem = nullptr;
    axcl_Malloc(&devMem, model_size, AXCL_MEM_MALLOC_NORMAL_ONLY, _devid);

    // 4. copy model to device
    axcl_Memcpy(devMem, model_buffer, model_size, AXCL_MEMCPY_HOST_TO_DEVICE, _devid);

    int ret = axcl_EngineLoadFromMem(devMem, model_size, &m_handle->handle, _devid);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    axcl_Free(devMem, _devid);
    // fprintf(stdout, "Engine creating handle is done.\n");

    return sub_init();
}

void ax_runner_ax650::release()
{
    if (m_handle && m_handle->handle)
    {
        free_io(&m_handle->io_data, _devid);
        axcl_EngineDestroyIO(m_handle->io, _devid);
        axcl_EngineUnload(m_handle->handle, _devid);
        m_handle->handle = 0;
    }

    if (m_handle)
    {
        delete m_handle;
        m_handle = nullptr;
    }

    moutput_tensors.clear();
    minput_tensors.clear();
    map_input_tensors.clear();
    map_output_tensors.clear();

    // AX_ENGINE_Deinit();
}

void ax_runner_ax650::deinit()
{
    if (m_handle && m_handle->handle)
    {
        // free_io(&m_handle->io_data);
        // mtensors.clear();
        // minput_tensors.clear();
        // map_input_tensors.clear();
        // map_tensors.clear();
        // AX_ENGINE_DestroyHandle(m_handle->handle);
        axcl_EngineDestroyIO(m_handle->io, _devid);
        axcl_EngineUnload(m_handle->handle, _devid);
        m_handle->handle = 0;
        // delete m_handle;
        // m_handle = nullptr;
    }

    // AX_ENGINE_Deinit();
}

int ax_runner_ax650::get_algo_width() { return -1; }
int ax_runner_ax650::get_algo_height() { return -1; }

int ax_runner_ax650::set_input(int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axcl_EngineSetInputBufferByIndex(m_handle->io, idx, (void *)phy_addr, size, _devid);
}
int ax_runner_ax650::set_output(int idx, unsigned long long int phy_addr, unsigned long size)
{
    return axcl_EngineSetOutputBufferByIndex(m_handle->io, idx, (void *)phy_addr, size, _devid);
}

ax_color_space_e ax_runner_ax650::get_color_space()
{
    // switch (m_handle->algo_colorformat)
    // {
    // case AX_FORMAT_RGB888:
    //     return ax_color_space_e::axdl_color_space_rgb;
    // case AX_FORMAT_BGR888:
    //     return ax_color_space_e::axdl_color_space_bgr;
    // case AX_FORMAT_YUV420_SEMIPLANAR:
    //     return ax_color_space_e::axdl_color_space_nv12;
    // default:
    //     return axdl_color_space_unknown;
    // }
    return axdl_color_space_unknown;
}

int ax_runner_ax650::inference(ax_image_t *pstFrame)
{
    // unsigned char *dst = (unsigned char *)minput_tensors[0].pVirAddr;
    // unsigned char *src = (unsigned char *)pstFrame->pVir;

    // switch (m_handle->algo_colorformat)
    // {
    // case AX_FORMAT_RGB888:
    // case AX_FORMAT_BGR888:
    //     for (size_t i = 0; i < pstFrame->nHeight; i++)
    //     {
    //         memcpy(dst + i * pstFrame->nWidth * 3, src + i * pstFrame->tStride_W * 3, pstFrame->nWidth * 3);
    //     }
    //     break;
    // case AX_FORMAT_YUV420_SEMIPLANAR:
    // case AX_FORMAT_YUV420_SEMIPLANAR_VU:
    //     for (size_t i = 0; i < pstFrame->nHeight * 1.5; i++)
    //     {
    //         memcpy(dst + i * pstFrame->nWidth, src + i * pstFrame->tStride_W, pstFrame->nWidth);
    //     }
    //     break;
    // default:
    //     break;
    // }

    // memcpy(minput_tensors[0].pVirAddr, pstFrame->pVir, minput_tensors[0].nSize);
    return inference();
}
int ax_runner_ax650::inference()
{
    // for (size_t i = 0; i < minput_tensors.size(); i++)
    // {
    //     axcl_Memcpy(
    //         (void *)minput_tensors[i].phyAddr, minput_tensors[i].nSize,
    //         minput_tensors[i].pVirAddr, minput_tensors[i].nSize, AXCL_MEMCPY_HOST_TO_DEVICE);
    // }
    auto ret = axcl_EngineExecute(m_handle->handle, m_handle->context, 0, m_handle->io, _devid);
    if (ret != 0)
    {
        ALOGE("AX_ENGINE_Execute");
        return ret;
    }

    // for (size_t i = 0; i < mtensors.size(); i++)
    // {
    //     axcl_Memcpy(
    //         mtensors[i].pVirAddr, mtensors[i].nSize,
    //         (void *)mtensors[i].phyAddr, mtensors[i].nSize, AXCL_MEMCPY_DEVICE_TO_HOST);
    // }
    return 0;
}

// int ax_cmmcpy(unsigned long long int dst, unsigned long long int src, int size)
// {
//     return AX_IVPS_CmmCopyTdp(dst, src, size);
// }