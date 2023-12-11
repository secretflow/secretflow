# Preprocessing Development Tips

Originally for each preprocessing component, we have a preprocessing_substituion component to apply the rule.

Now we introduce a unfiied preprocessing_component `substitution` that can be used in place of the original preprocessing_substitution component.

## Idea

During preprocessing

1. we dump the trace runner objects (see sc.compute)
2. we dump the meta change with it.

During the substitution phase:

1. We load the trace runner objects from the dumped files and use them to do the substitution.
2. We load the meta change from the dumped file and apply it.
