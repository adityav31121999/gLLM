
#include <iostream>
#include <vector>
#include <string>
#include "gllm.h"

#ifdef USE_OPENCL
    // Source files - Paths relative to this header file's location
    std::vector<std::string> kernelSourceFiles = {
        "../../../src/maths/src/mat/cl/operators.cl",
        "../../../src/neural/src/attention/cl/attention.cl",
        "../../../src/neural/src/mlp/cl/mlp.cl",
        "../../../src/neural/src/transformer/cl/transformer.cl",
        "../../../src/maths/src/basic/cl/activations.cl",
        "../../../src/maths/src/basic/cl/vect.cl"
    };

    // Kernel names - Must be unique across ALL source files loaded into the context
    std::vector<std::string> kernelNames = {
        // from maths -> vect.cl
        "matrixMultiplyKernel",
        "vectorAddKernel",

        // from maths -> operators.cl
        "dot_matrix_vector",
        "dot_vector_matrix_vector",

        // from maths -> activations.cl
        "clSigmoid",
        "clSigmoid1d",
        "clSigmoid2d",
        "clSigmoidder",
        "clSigmoid1dder",
        "clSigmoid2dder",
        // "parallel_reduce_max",
        // "parallel_reduce_sum",
        "clSoftmax1d",
        "clSoftmax2d",
        "clSoftmax1dder",
        "clSoftmax2dder",
        "clSoftmaxd1dder_from_s",
        "clSoftmaxd2dder_from_s",
        "clReLU",
        "clReLU1d",
        "clReLU2d",
        "clReLUder",
        "clReLUder1d",
        "clReLUder2d",
        // "parallel_reduce_min",
        "clLOTA1d",
        "clLOTA2d",
        "clLOTA2dmasking",
        "clLOTA1dder",
        "clLOTA2dder",
        "clLOTA2ddermasking",

        // from mlp.cl
        "l1PenaltyKernel",
        "l2PenaltyKernel",
        "absDiffKernel",
        "squaredDiffKernel",
        "kernelComputeGradMLPInput",
        "kernelOutputDeltaSigmoid",
        "kernelHiddenDeltaSigmoid",
        "kernelLastLayerDeltaSigmoid",
        "kernelUpdateWeightsAndGradients",
        "kernelUpdateWeights",
        "kernelUpdateWeightsL1",
        "kernelUpdateWeightsL2",
        "kernelUpdateInputMLP",
        "kernelLayerForward",
        "kernelMseReduction",
        "kernelRpropUpdate",

        // from neural -> attention.cl
        "vectorAddKernel_attention",
        "kernelVecDotVec",
        "kernelCompute_single_kq_vector",
        "kernelComputeKorQ",
        "kernelDotvecmatvec",
        "kernelComputePrediction",
        "kernelElementwiseMultiply",
        // "atomic_add_float",
        "kernelKdotQforSelf_train",
        "kernelKdotQforCross_train",
        "kernelKdotQ_Block1_Self_Inference",
        "kernelKdotQ_Block1_Cross_Inference",
        "kernelKdotQ_BlockN_Self_Inference",
        "kernelKdotQ_BlockN_Cross_Inference",
        "computeHeadSumsMaskedKernel",
        "accumulateWeightedVectorsKernel",
        "accumulateEVRowsKernelCL",
        "updateEVRowsKernelCL",
        "kernelComputeGradientsEH",
        "kernelComputeGradDhDv_1stHead",
        "kernelComputeGradientsEH_EV",
        "kernelComputeGradDhDv",
        "kernelComputePreMH_MV",
        "kernelComputeGradMH_MV",
        "kernelComputeGradHead",
        "kernelComputeGradKdotQ_LOTA",
        "kernelComputeGradK_Q",
        "kernelComputeGradMK_MQ",
        "kernelUpdateWeights_EH_EV",
        "kernelComputeGradientsEV_V",
        "kernelComputeGradDv_V",
        "kernelComputePreMV_V",
        "kernelComputeGradMV_V",
        "kernelComputeGradHead_V",
        "kernelComputeGradQ_V",
        "kernelComputeGradMQ_V",
        "kernelComputeGradMKCorrection",
        "kernelUpdateWeights_EV_V",
        "kernelComputeSimpleLOTAder",
        "kernelRowSum",
        "kernelComputeGradMK_MQ_Simplified",
        "kernelUpdateWeights_1stHead_H",
        "kernelUpdateWeights_1stHead_V",
        "kernelUpdateWeights_1stHead_HV",
        "kernelUpdateSimple",
        
        // from transformer/transformer.cl
        "kernelKdotQforSelf_train_transformer",
        "kernelKdotQforCross_train_transformer",
        "kernelKdotQBlock1Self_Inference",
        "kernelKdotQBlock1Cross_Inference",
        "kernelKdotQBlockNSelf_Inference",
        "kernelKdotQBlockNCross_Inference",
    };
#endif
