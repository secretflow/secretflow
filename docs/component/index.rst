.. _component:

Component
=========

SecretFlow Component is based on two Specifications. On the one hand, we obey `SecretFlow Open Specification <https://www.secretflow.org.cn/docs/spec/latest>`_. On the other, we defined cluster config and
some other data types with Extended Specification at :doc:`/component/comp_spec`. Extended Specification only applies on SecretFlow and may not used in other privacy-preserving applications in SecretFlow ecosystem.
The detailed explanation of Extended Specification is at :doc:`/component/comp_spec_design`.

We wrapped some commonly used SecretFlow applications as SecretFlow Component List. However, it is not final and we are updating the list promptly. The full SecretFlow Component List is available at :doc:`/component/comp_list`.

Now, you could run SecretFlow applications with component API or CLI apart from writing Python programs. Please check :doc:`/component/comp_guide`.



Migration to `SecretFlow Open Specification <https://www.secretflow.org.cn/docs/spec/latest>`_
-----------------------------------------------------------------------------------------------

There are some breaking changes after introduction of **SecretFlow Open Specification**, including

comp.proto
^^^^^^^^^^

1. *comp.proto* is renamed to *component.proto*
2. In message *AttrType*, *AT_UNDEFINED* is replaced with *ATTR_TYPE_UNSPECIFIED*
3. In message *Attribute*, *has_lower_bound* is renamed to *lower_bound_enabled* and *has_upper_bound* is renamed to *upper_bound_enabled*
4. In message *IoDef.TableAttrDef*, *attrs* is renamed to *extra_attrs*

data.proto
^^^^^^^^^^

1. In message *SystemInfo*, *app_name* is renamed to *app* while *secretflow* (SFClusterDesc) is replaced with *app_meta* (Any).
2. Message *StorageConfig* is moved from *cluster.proto*
3. In message *IndividualTable* and *VerticalTable*, *num_lines* is renamed to *line_count*.
4. In message *DistData*, *sys_info* is renamed to *system_info*.

We are sorry for inconvenience.

Announcement
------------

October, 2023
^^^^^^^^^^^^^

1. We officially launched `SecretFlow Open Specification <https://www.secretflow.org.cn/docs/spec/latest>`_ which contains the most common parts which are shared by other privacy-preserving applications.
2. The remaining part are called Extended Specification.



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
