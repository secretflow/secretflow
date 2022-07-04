:notoc:

Secretflow
=============

`Secretflow <https://github.com/secretflow/secretflow>`_ is an unified framework for privacy-preserving data analysis and machine learning.

Secretflow provides

- Device abstraction, which abstracts privacy computing technologies such as 
  Multi-Party Secure Computing (MPC), Homomorphic Encryption (HE), 
  and Trusted Execution Environment (TEE) into ciphertext devices, 
  and abstracts plaintext computing into plaintext devices.
- Computational graphs based on abstracted devices, enabling data analysis
  and machine learning workflows to be represented as computational graphs.
- Machine learning/data analysis capabilities based on computational graphs,
  supporting data horizontal/vertical/hybrid segmentation and other scenarios.

.. image:: _static/secretflow_arch.svg

Why `Secretflow`
----------------

At present, privacy computing technology is growing in popularity. 
However, neither the technology nor the market has yet reached real maturity.
In order to cope with the development uncertainty of privacy computing technology and applications, 
we propose a general privacy computing framework called "secretflow".
Secretflow will adhere to the following principles, 
so that the framework has the maximum inclusive and extensible capabilities to 
cope with the development of future privacy computing technologies and applications.

- Completeness: It supports various privacy computing technologies and 
  can be assembled flexibly to meet the needs of different scenarios.
- Transparency: Build a unified technical framework, and try to make the 
  underlying technology iteration transparent to the upper-layer application, 
  with high cohesion and low coupling.
- Openness: People with different professional directions can easily participate
  in the construction of the framework, and jointly accelerate the development of privacy computing technology.
- Connectivity: Data in scenarios supported by different underlying technologies
  can be connected to each other.

Documents
----------
.. panels::

    ---

    Getting started
    ^^^^^^^^^^^^^^^

    New to **Secretflow**? Check out the getting stgarted guides.

    +++
    
    .. link-button:: getting_started
            :type: ref
            :text: To the getting started guides
            :classes: btn-primary stretched-link

    ---

    Toturials
    ^^^^^^^^^^^^^^^

    Want to know in-depth usages of **Secretflow**? Check out our toturials.

    +++
    
    .. link-button:: tutorial
            :type: ref
            :text: To the toturials
            :classes: btn-primary stretched-link

    ---

    API reference
    ^^^^^^^^^^^^^^^

    Get a detailed description of the secretflow api.

    +++
    
    .. link-button:: reference
            :type: ref
            :text: To the reference guide
            :classes: btn-primary stretched-link

    ---

    Developer guide
    ^^^^^^^^^^^^^^^

    Want to know the design principles of secretflow?
    Want to improve existing functions or add more functions?
    The developer guidelines will guide you.
    

    +++
    
    .. link-button:: development
            :type: ref
            :text: To the development guide
            :classes: btn-primary stretched-link

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started/index
   tutorial/index
   reference/index
   development/index
