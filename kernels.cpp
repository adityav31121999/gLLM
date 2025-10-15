
#include <iostream>
#include <vector>
#include <string>
#include "gllm.h"

#ifdef USE_OPENCL

    // Source files - Paths relative to this header file's location
    std::vector<std::string> kernelSourceFiles = {
        "d:/gLLM/src/maths/src/basic/cl/activations.cl",
        "d:/gLLM/src/maths/src/basic/cl/vect.cl",
        "d:/gLLM/src/maths/src/mat/cl/operators.cl",
        "d:/gLLM/src/neural/src/mlp/cl/mlp.cl",
        "d:/gLLM/src/neural/src/attention/cl/attention.cl",
        "d:/gLLM/src/neural/src/attention/cl/kdotq.cl",
        "d:/gLLM/src/neural/src/attention/cl/weights.cl",
        "d:/gLLM/src/model/src/tokens/token.cl"
    };

    // Kernel names - Must be unique across ALL source files loaded into the context
    std::vector<std::string> kernelNames = {
        // vect.cl
        "matrixMultiplyKernel",
        "vectorAddKernel",
        "vectorsAddKernel",
        // operators.cl
        "kernelTransposeMatrix",
        "matrix_multiply",
        "vector_matrix_multiply",
        "dot_matrix_vector",
        "vectorxMatTkernel",
        "matxMatTkernel",
        "dot_vector_matrix_vector",
        // activations.cl
        "clSigmoid",
        "clSigmoid1d",
        "clSigmoid2d",
        "clSigmoidder",
        "clSigmoid1dder",
        "clSigmoid2dder",
        "clSoftmax1d",
        "clSoftmax2d",
        "clSoftmax1dder",
        "clSoftmax2dder",
        "clReLU",
        "clReLU1d",
        "clReLU2d",
        "clReLUder",
        "clReLUder2d",
        "clLOTA1d",
        "clLOTA2d",
        "clLOTA2dmasking",
        "clLOTA1dder",
        "clLOTA2dder",
        "clLOTA2ddermasking",
        // mlp.cl
        "kernelOutputDelta",
        "l1PenaltyKernel",
        "l2PenaltyKernel",
        "absDiffKernel",
        "squaredDiffKernel",
        "kernelComputeGradMLPInput",
        "kernelOutputDeltaSigmoid",
        "kernelHiddenDeltaSigmoid",
        "kernelLastLayerDeltaSigmoid",
        "kernelUpdateWeights",
        "kernelUpdateWeightsAndGradients",
        "kernelUpdateInputMLP",
        "kernelLayerForward",
        "kernelMseReduction",
        "kernelRpropUpdate",
        "kernelUpdateElasticNet",
        // attention.cl
        "updateEVRowsKernelCL",
        "kernelElementwiseMultiply",
        "kernelComputeHeadSumsMasked",
        "kernelComputeHeadSumsMaskedev",
        "kernelAccumulateWeightedVectors",
        "kernelAccumulateWeightedVectorsev",
        "kernelComputeGradpred",
        "KernelComputeGradDeEmbeddings",
        "kernelGradForAttentionOutput",
        "kernelComputeGradientsEH",
        "kernelComputeGradientsEH_EV",
        "kernelComputeGradientsEHEVFromMSE",
        "kernelComputeGradientsEV_V",
        "kernelComputeGradDhDv",
        "kernelComputeGradDhDv_1stHead",
        "kernelComputePreMH_MV",
        "kernelComputeGradMH_MV",
        "kernelComputeGradHead",
        "kernelComputeGradKdotQ_LOTA",
        "kernelComputeGradK_Q",
        "kernelComputeGradMK_MQ",
        "kernelComputePreMV_V",
        "kernelComputeGradMV_V",
        "kernelComputeGradHead_V",
        "kernelComputeGradQ_V",
        "kernelComputeGradMQ_V",
        "kernelComputeGradMKCorrection",
        "kernelComputeSimpleLOTAder",
        "kernelRowSum",
        "kernelComputeGradMK_MQ_Simplified",
        // kdotq.cl
        "kernelCompute_single_kq_vector",
        "kernelComputeKQall",
        "kernelKdotQforSelf_train",
        "kernelKdotQforCross_train",
        "kernelKdotQ_Block1_Selfi",
        "kernelKdotQ_Block1_Crossi",
        "kernelKdotQ_BlockN_Selfi",
        "kernelKdotQ_BlockN_Crossi",
        // weights.cl
        "kernelUpdateEVrows",
        "kernelUpdateWeightsHeadHVElastic",
        "kernelUpdateWeightsHeadElastic",
        "kernelUpdateWeightsGeneral",
        "kernelUpdateWeightsGeneral_f4",
        "kernelUpdateSimple",
        "kernelUpdateSimple_Elastic",
        "accumulateEVRowsKernelCL",
        "kernelComputePrediction",
        "kernelComputePredictionWithScores",
        "updateEmbeddings",
        // model
        "generate_embeddings",
        "batchedVectorInverseKernel"
    };

#endif

/*
                if(current_error < prev_error) {
                    if(j <= 6)   
                        learning *= 1.1;
                    else if (j % 6 == 0)
                        learning *= (1.05 + (j/6)*0.15);
                }
                else {
                    if(j <= 6)   
                        learning *= 0.95;
                    else if (j % 6 == 0)
                        learning *= (0.95 - (j/6)*0.01);
                }

*/