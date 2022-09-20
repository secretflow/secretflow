HEU Device
==========

What is HEU
-----------

Homomorphic Encryption processing Unit (HEU) is a submodule of Secretflow that implements high-performance homomorphic encryption algorithms.

The purpose of HEU is to lower the threshold for the use of homomorphic encryption, so that users can use homomorphic encryption algorithms to build privacy-preserving applications without professional cryptography knowledge.


The Design of HEU
-----------------

.. raw:: html

    <style>
        .red {color:#CC6600}
        .blue {color:#6C8EBF}
    </style>

.. role:: red
.. role:: blue

HEU has two meanings, it is both a device in Secretflow and a high-performance homomorphic encryption library that can be used independently.

.. image:: img/heu_layer.svg
   :align: center

:red:`HEU Library`: You can view HEU as a high-performance and complete homomorphic encryption library, which integrates almost all homomorphic encryption algorithms in the industry. At the same time, HEU encapsulates each algorithm and provides a uniform interface called "HEU library access layer". You can switch between different HE algorithms at any time without modifying business code.

:blue:`HEU device`: As a component of Secretflow, HEU abstracts the homomorphic encryption algorithms into a programmable device, making it easy for users to flexibly build applications using the homomorphic encryption technology without professional knowledge. HEU (device) aims to build a complete computing solution through HE, that is, based on HE, any type of computing can be completed. Compared with SPU, HEU's computation is purely local without any network communication, so HEU and SPU are complementary


Insight into HEU
^^^^^^^^^^^^^^^^

Now let's see the detail of HEU library.

There are many kinds of homomorphic encryption algorithms, and the functions of different algorithms are very different. It is impossible to integrate all the algorithms into a same library and provide a uniform interface. Therefore, HEU classifies homomorphic encryption algorithms into three categories, corresponding to three different working modes, as shown in the following table:

.. csv-table:: HEU Working Mode
   :header: "Working Mode", "Supported Calculation Types", "Number Of Calculations", "HE Algorithms", "Calculating Speed", "Ciphertext Size"

    "PHEU", "Addition", "Unlimited", "Paillier", "Fast", "Small"
    "LHEU", "Addition, Multiplication", "Limited", "BGV, CKKS", "Fast (packed Mode)", "Least (packed Mode)"
    "FHEU", "Addition, Multiplication, Comparison, Mux", "Unlimited", "TFHE (Bitwise)", "Very Slow", "Largest"

.. note:: HEU is still under rapid iterative development, LHEU and FHEU modes are not ready currently, only PHEU mode is available.

Based on the three working modes of the HEU, the architecture of the HEU library is divided into three relatively independent parts in vertical direction.

.. image:: img/heu_arch.png

The left part integrates PHE-related libraries, including Paillier, ElGamal, and so on. HEU deeply optimizes these PHE algorithms and often performs better than third-party libraries. So even if you do not need to use SecretFlow, it's beneficial to just use HEU alone as a PHE library.

The middle part integrates LHE libraries, including SEAL, HELib, Palisade, and so on. It should be noted that although BGV/BFV and CKKS are fully homomorphic encryption algorithms themselves, the performance of bootstrapping is very low, so HEU only uses them as leveled HEs.

The right part integrates the FHE library, namely `Concrete`_, which is a variant of the TFHE algorithm developed by `Zama.ai`_. It only supports encrypt one bit at a time, so its performance is very low. For example, a 64-bit integer will generate 64 ciphertexts after encrypted in this mode. And if you want to add two 64-bit integers, you need to execute a very complex circuit to obtain the result, which is very inefficient. In short, it is only suitable for low computation cost scenarios, and the advantage is that it can support arbitrary kind of encrypted operations, making it possible to do confidential outsourcing computing.

.. _Concrete: https://github.com/zama-ai/concrete
.. _Zama.ai: https://www.zama.ai/

From a horizontal perspective, HEU can also be divided into several layers. The top layer is the ``LLO`` (low level operators) layer, which is the entry point to the HEU library. The HEU device first translates the user's code into a DAG, and then compiled into LLO code. LLO defines a atomic operations set for interworking with HEU library. The second layer is ``Rewriter``, which translates and rewrites the LLO code to suit different working modes. The third layer is ``Accelerator``, which further optimizes the LLO code to accelerate the performance of confidential computing. The next layer is ``Access Layer``, which abstracts a uniform interface for different homomorphic encryption libraries to facilitate protocol insertion and removal. The penultimate layer ``Protocol`` layer, which implements a variety of specific homomorphic encryption algorithms. The bottom layer is ``Hardware`` acceleration layer, which is still under working. We plan to support PHE hardware acceleration first, and then further expand to LHE and FHE.
