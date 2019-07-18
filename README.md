# TernaryCNN

The python file contains our Pytorch implementation of the Gated-XNOR architecture for the MNIST dataset.

The header files of each layer implement each layer with customizable parameters for a base processor. The full_v2.c includes the forward pass implementation in C using the each of the header files implemented on a configurable processor.

The TIEfiles folder contains the Verilog like TIE progam files which define the custom instruction. The NN_with_tie folder contains code that implement each layer using the custom instructions and extra state registers.
