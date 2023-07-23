.. _component:

Component
=========

.. Note:: Both Component Specification and SecretFlow Component List are subject to modify at this moment.

Component Specification is a protocol brought by SecretFlow Ecosystem. It mainly includes Data, Component and Node Evaluation protocol.
Logically, if you wrap any program with Component Specification, it will be recognized by Kuscia(Deployment and scheduling of SecretFlow stack) and
SecretFlow Platform [#f1]_ (User Interface and Control Panel) . In the one word, for system developers, if you wrap your application with Component Specification,
you could utilize other open-source SecretFlow projects with little effort.

Based on Component Specification, we wrapped some commonly used SecretFlow applications. The SecretFlow Component List is not final,
we are updating the list promptly. Now, you could run SecretFlow applications with component API or CLI apart from writing Python programs.

Besides SecretFlow, we are going to wrap other applications including SCQL and TECC with Component Specification.

For developers who are not familiar with :doc:`/component/comp_spec`, they are encouraged to check :doc:`/component/comp_spec_design` first.

After that, please check :doc:`/component/comp_list` and corresponding guides at :doc:`/component/comp_guide`.


.. rubric:: Footnotes

.. [#f1] Product name is undecided.



Announcement
------------

July, 2023
^^^^^^^^^^

From SecretFlow 1.0.0, we officially launch the first version of Component Specification and SecretFlow Component List based on it.




.. toctree::
   :maxdepth: 1
   :caption: References

   comp_spec
   comp_list

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   comp_spec_design
   comp_guide
