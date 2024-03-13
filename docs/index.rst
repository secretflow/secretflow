:notoc:

SecretFlow
=============

`SecretFlow <https://github.com/secretflow/secretflow>`_ is a unified framework for privacy-preserving data analysis and machine learning.

SecretFlow provides

- Device abstraction, which abstracts privacy-preserving computing technologies such as
  Multi-Party Secure Computing (MPC), Homomorphic Encryption (HE),
  and Trusted Execution Environment (TEE) into ciphertext devices,
  and abstracts plaintext computing into plaintext devices.
- Computational graphs based on abstracted devices, enabling data analysis
  and machine learning workflows to be represented as computational graphs.
- Machine learning/data analysis capabilities based on computational graphs,
  supporting data horizontal/vertical/hybrid segmentation and other scenarios.

.. image:: _static/secretflow_arch.svg

Why `SecretFlow`
----------------

At present, privacy-preserving computing technology is growing in popularity.
However, neither the technology nor the market has yet reached real maturity.
In order to cope with the development uncertainty of privacy-preserving computing technology and applications,
we propose a general privacy-preserving computing framework called "SecretFlow".
SecretFlow will adhere to the following principles,
so that the framework has the maximum inclusive and extensible capabilities to
cope with the development of future privacy-preserving computing technologies and applications.

- Completeness: It supports various privacy-preserving computing technologies and
  can be assembled flexibly to meet the needs of different scenarios.
- Transparency: Build a unified technical framework, and try to make the
  underlying technology iteration transparent to the upper-layer application,
  with high cohesion and low coupling.
- Openness: People with different professional directions can easily participate
  in the construction of the framework, and jointly accelerate the development of privacy-preserving computing technology.
- Connectivity: Data in scenarios supported by different underlying technologies
  can be connected to each other.


MPC, FL, TEE: which one is better?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We're often asked which technology is better. The short answer is there is no dominant winner.
It depends on the requirements, such as performance, security assumption, etc. For a more detailed comparison, please check
`A comprehensive comparison of various privacy-preserving technologies <https://www.yuque.com/secret-flow/admin/exgixt72drdvdsy3>`_ .


.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started/index
   user_guide/index
   api/index
   tutorial/index
   component/index
   developer/index
