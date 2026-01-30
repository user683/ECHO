Add models with the Megatron-LM backend
=========================================

Last updated: 04/25/2025.

Model
-----------


If use latest verl, we have direct support of ``GPTModel`` for Megatron backend. 
You can use the similar way of using Megatron to pretrain custom models. 
We list the steps here:

2. If your model is configurable by ``TransformerLayerSpec`` , you can
   directly use ``GPTModel``. Otherwise, Please implement a new
   ``ModelLayerSpec`` and ``ModelLayer`` here.
   as arguments to initialize the GPTModel.
4. Return the model at last.
