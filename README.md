
# gLLM - AI/ML Library for creating LLMs

## INTRO
- Library for LLM Models
- **VERSION**: 0.0.0.1

## Project Structure
- **maths**: Mathematical Library for AI/ML
- **neural**: Neural Network Library
- **model**: Model Library

### Root Directory


### src/maths


### src/neural


### src/model


### src/script


### bin
- Output directory for compiled binaries

## Library Details


## Third-Party Dependencies
- **OpenCL**: Used for GPU acceleration

## MODEL STRUCTURE
#MODEL METADATA#
- FILE EXTENSION: .gen

```
                               ___
                              /      :                       
                              |--> NeuNet <-- Transformer(Network) <-----------<----> Input and Prompt
                              |      :                                        /
           ____               |-->   :                                        |<----> Token Formation 
          /                   |      :                                        |
          |                   |--> NeuNet <-- Transformer(Network)            |<----> Propagations
          |                   |                                               |
          |       Data        |--> NeuNet <-- Transformer(Network)            |<----> Feature Extraction
          | Trained Networks  |                                               \
Model <-->|    Parameters  <--|--> NeuNet <-- Transformer(Network)             <----> Prediction & Output
          |      Weights      |
          |   Communication   |--> NeuNet <-- Transformer(Network)
          |                   |
          |                   |--> NeuNet <-- Transformer(Network)
          |                   |      :
          \____               |-->   :
                              |      :
                              |--> NeuNet <-- Transformer(Network)
                              \___   :

```

### Transformer Structure

```

```
