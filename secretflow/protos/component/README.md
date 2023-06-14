**NOTE: component is still experimental.**

This is a high level overview of data stuctures related to component.

* cluster: First step, we should setup a SecretFlow cluster
    - **SFClusterDesc**: Description of a SecretFlow cluster. Two cluster with the same **SFClusterDesc** could be regarded as the same since they have the same topology and security settings.
    - **SFClusterConfig**: Config for setting up a SecretFlow cluster. It contains **SFClusterDesc** and other runtime settings for network and storage.

* data: Data for SecretFlow Applications, e.g. SecretFlow/SCQL
    - **SystemInfo**: Describe the application which could consume the data.
    - **DistData**
        - **DataRef**: **DataRef** refers to a piece of solid data in the SecretFlow cluster. A **DistData** contains a list of **DataRef**s
        - meta: The meta would contains the extra to describe how **DataRef**s is organized. Now we support the following first-class Data.
            - **VerticalTable**: Vertical table.

* comp: Definitations for a component.
    - **ComponentDef**: A component consists of name, attributes, inputs and outputs.
    - **CompListDef**: A list of components.
    - **AttributeDef**: Defines a attribute. We organize the attributes of a component as a tree. The leaves of the tree is called *Atomic Attribute*, which represent a solid field need to fill. The non-leaf nodes are *Union Attribute Group* and *Struct Attribute Group*. The children of a *Struct Attribute Group* node could be regarded as a group. While for children of a *Union Attribute Group*, user should select one child to fill.
        - **AtomicAttrDesc**: Extra settings for a *Atomic Attribute*
        - **UnionAttrGroupDesc** Extra settings for a *Union Attribute Group*
    - **IoDef**: Defines an input/output of a component.

* evaluation: A call to a component.
    - **NodeEvalParam**: All the info to evaluate a component - id to the component, solid attributes and inputs, output names.
    - **NodeEvalResult**: The result of evaluation - solid output.
