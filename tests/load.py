# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import cycle

from _pytest.runner import CollectReport
from xdist.remote import Producer
from xdist.report import report_collection_diff
from xdist.workermanage import parse_spec_config

# Adapt from https://github.com/pytest-dev/pytest-xdist/blob/master/src/xdist/scheduler/load.py

SF_PARTY_PREFIX = 'SF_PARTY_PREFIX_'

SF_PARTIES = [
    SF_PARTY_PREFIX + 'alice',
    SF_PARTY_PREFIX + 'bob',
    SF_PARTY_PREFIX + 'carol',
    SF_PARTY_PREFIX + 'davy',
]


# High level idea: each node would only be act as one party of secretflow cluster.
class SFLoadPartyScheduling:
    """Implement load scheduling across nodes.

    This distributes the tests collected across all nodes so each test
    is run just once.  All nodes collect and submit the test suite and
    when all collections are received it is verified they are
    identical collections.  Then the collection gets divided up in
    chunks and chunks get submitted to nodes.  Whenever a node finishes
    an item, it calls ``.mark_test_complete()`` which will trigger the
    scheduler to assign more tests if the number of pending tests for
    the node falls below a low-watermark.

    When created, ``numnodes`` defines how many nodes are expected to
    submit a collection. This is used to know when all nodes have
    finished collection or how large the chunks need to be created.

    Attributes:

    :numnodes: The expected number of nodes taking part.  The actual
       number of nodes will vary during the scheduler's lifetime as
       nodes are added by the DSession as they are brought up and
       removed either because of a dead node or normal shutdown.  This
       number is primarily used to know when the initial collection is
       completed.

    :node2collection: Map of nodes and their test collection.  All
       collections should always be identical.

    :node2pending: Map of nodes and the indices of their pending
       tests.  The indices are an index into ``.pending`` (which is
       identical to their own collection stored in
       ``.node2collection``).

    :collection: The one collection once it is validated to be
       identical between all the nodes.  It is initialised to None
       until ``.schedule()`` is called.

    :pending: List of indices of globally pending tests.  These are
       tests which have not yet been allocated to a chunk for a node
       to process.

    :log: A py.log.Producer instance.

    :config: Config object, used for handling hooks.
    """

    def __init__(self, config, log=None):
        self.numnodes = len(parse_spec_config(config))
        assert self.numnodes >= len(SF_PARTIES)
        self.node2collection = {}
        self.node2pending = {}
        self.pending = set()
        self.collection = None
        if log is None:
            self.log = Producer("sfloadpartysched")
        else:
            self.log = log.loadsched
        self.config = config
        self.maxschedchunk = self.config.getoption("maxschedchunk")
        # added
        self.exclusive = [[] for _ in range(len(SF_PARTIES))]
        self.nonexclusive = set()
        self.node2party = {}

    def _organize_test(self):
        indexed_collection = list(enumerate(self.collection))
        sorted_indexed_collection = sorted(indexed_collection, key=lambda x: x[1])

        for idx, test in sorted_indexed_collection:
            is_exclusive = False
            for i, p in enumerate(SF_PARTIES):
                if test.find(p) != -1:
                    is_exclusive = True
                    self.exclusive[i].append(idx)
                    break
            if not is_exclusive:
                self.nonexclusive.add(idx)

    def _add_node2party(self, node):
        if len(self.node2party) < len(SF_PARTIES):
            self.node2party[node] = len(self.node2party)

    @property
    def nodes(self):
        """A list of all nodes in the scheduler."""
        return list(self.node2pending.keys())

    @property
    def collection_is_completed(self):
        """Boolean indication initial test collection is complete.

        This is a boolean indicating all initial participating nodes
        have finished collection.  The required number of initial
        nodes is defined by ``.numnodes``.
        """
        return len(self.node2collection) >= self.numnodes

    @property
    def tests_finished(self):
        """Return True if all tests have been executed by the nodes."""
        if not self.collection_is_completed:
            return False
        if self.pending:
            return False
        for pending in self.node2pending.values():
            if len(pending) >= 2:
                return False
        return True

    @property
    def has_pending(self):
        """Return True if there are pending test items

        This indicates that collection has finished and nodes are
        still processing test items, so this can be thought of as
        "the scheduler is active".
        """
        if self.pending:
            return True
        for pending in self.node2pending.values():
            if pending:
                return True
        return False

    def add_node(self, node):
        """Add a new node to the scheduler.

        From now on the node will be allocated chunks of tests to
        execute.

        Called by the ``DSession.worker_workerready`` hook when it
        successfully bootstraps a new node.
        """
        assert node not in self.node2pending
        self.node2pending[node] = []
        self._add_node2party(node)

    def add_node_collection(self, node, collection):
        """Add the collected test items from a node

        The collection is stored in the ``.node2collection`` map.
        Called by the ``DSession.worker_collectionfinish`` hook.
        """
        assert node in self.node2pending
        if self.collection_is_completed:
            # A new node has been added later, perhaps an original one died.
            # .schedule() should have
            # been called by now
            assert self.collection
            if collection != self.collection:
                other_node = next(iter(self.node2collection.keys()))
                msg = report_collection_diff(
                    self.collection, collection, other_node.gateway.id, node.gateway.id
                )
                self.log(msg)
                return
        self.node2collection[node] = list(collection)

    def mark_test_complete(self, node, item_index, duration=0):
        """Mark test item as completed by node

        The duration it took to execute the item is used as a hint to
        the scheduler.

        This is called by the ``DSession.worker_testreport`` hook.
        """
        self.node2pending[node].remove(item_index)
        self.check_schedule(node, duration=duration)

    def mark_test_pending(self, item):
        raise NotImplementedError()
        # self.pending.insert(
        #     0,
        #     self.collection.index(item),
        # )
        # for node in self.node2pending:
        #     self.check_schedule(node)

    def check_schedule(self, node, duration=0):
        """Maybe schedule new items on the node

        If there are any globally pending nodes left then this will
        check if the given node should be given any more tests.  The
        ``duration`` of the last test is optionally used as a
        heuristic to influence how many tests the node is assigned.
        """
        if node.shutting_down:
            return

        if self.pending:
            # how many nodes do we have?
            num_nodes = len(self.node2pending)
            # if our node goes below a heuristic minimum, fill it out to
            # heuristic maximum
            items_per_node_min = max(2, len(self.pending) // num_nodes // 4)
            items_per_node_max = max(2, len(self.pending) // num_nodes // 2)
            node_pending = self.node2pending[node]
            if len(node_pending) < items_per_node_min:
                if duration >= 0.1 and len(node_pending) >= 2:
                    # seems the node is doing long-running tests
                    # and has enough items to continue
                    # so let's rather wait with sending new items
                    return
                num_send = items_per_node_max - len(node_pending)
                # keep at least 2 tests pending even if --maxschedchunk=1
                maxschedchunk = max(2 - len(node_pending), self.maxschedchunk)
                self._send_tests(node, min(num_send, maxschedchunk))
        else:
            node.shutdown()

        self.log("num items waiting for node:", len(self.pending))

    def remove_node(self, node):
        """Remove a node from the scheduler

        This should be called either when the node crashed or at
        shutdown time.  In the former case any pending items assigned
        to the node will be re-scheduled.  Called by the
        ``DSession.worker_workerfinished`` and
        ``DSession.worker_errordown`` hooks.

        Return the item which was being executing while the node
        crashed or None if the node has no more pending items.

        """
        pending = self.node2pending.pop(node)
        if not pending:
            return

        # The node crashed, reassing pending items
        crashitem = self.collection[pending.pop(0)]
        self.pending.update(pending)
        for node in self.node2pending:
            self.check_schedule(node)
        return crashitem

    def schedule(self):
        """Initiate distribution of the test collection

        Initiate scheduling of the items across the nodes.  If this
        gets called again later it behaves the same as calling
        ``.check_schedule()`` on all nodes so that newly added nodes
        will start to be used.

        This is called by the ``DSession.worker_collectionfinish`` hook
        if ``.collection_is_completed`` is True.
        """
        assert self.collection_is_completed
        assert len(self.nodes) >= len(SF_PARTIES)

        # Initial distribution already happened, reschedule on all nodes
        if self.collection is not None:
            for node in self.nodes:
                self.check_schedule(node)
            return

        # XXX allow nodes to have different collections
        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return

        # Collections are identical, create the index of pending items.
        self.collection = list(self.node2collection.values())[0]
        # self.pending[:] = range(len(self.collection))
        self.pending.update(range(len(self.collection)))
        self._organize_test()
        if not self.collection:
            return

        if self.maxschedchunk is None:
            self.maxschedchunk = len(self.collection)

        # Send a batch of tests to run. If we don't have at least two
        # tests per node, we have to send them all so that we can send
        # shutdown signals and get all nodes working.
        if len(self.pending) < 2 * len(self.nodes):
            # Distribute tests round-robin. Try to load all nodes if there are
            # enough tests. The other branch tries sends at least 2 tests
            # to each node - which is suboptimal when you have less than
            # 2 * len(nodes) tests.
            nodes = cycle(self.nodes)
            for i in range(len(self.pending)):
                self._send_tests(next(nodes), 1)
        else:
            # Send batches of consecutive tests. By default, pytest sorts tests
            # in order for optimal single-threaded execution, minimizing the
            # number of necessary fixture setup/teardown. Try to keep that
            # optimal order for every worker.

            # how many items per node do we have about?
            items_per_node = len(self.collection) // len(self.node2pending)
            # take a fraction of tests for initial distribution
            node_chunksize = min(items_per_node // 4, self.maxschedchunk)
            node_chunksize = max(node_chunksize, 2)
            # and initialize each node with a chunk of tests
            for node in self.nodes:
                self._send_tests(node, node_chunksize)

        if not self.pending:
            # initial distribution sent all tests, start node shutdown
            for node in self.nodes:
                node.shutdown()

    def _send_tests(self, node, num):
        # tests_per_node = self.pending[:num]
        tests_per_node = []
        if node in self.node2party:
            party = self.node2party[node]
            for i in range(min(num, len(self.exclusive[party]))):
                tests_per_node.append(self.exclusive[party].pop(0))
            num -= len(tests_per_node)

        for i in range(min(num, len(self.nonexclusive))):
            tests_per_node.append(self.nonexclusive.pop())

        for test in tests_per_node:
            self.pending.remove(test)

        # for i in range(min(num, len(self.pending))):
        #     tests_per_node.append(self.pending.pop())
        if tests_per_node:
            # del self.pending[:num]
            self.node2pending[node].extend(tests_per_node)
            node.send_runtest_some(tests_per_node)

    def _check_nodes_have_same_collection(self):
        """Return True if all nodes have collected the same items.

        If collections differ, this method returns False while logging
        the collection differences and posting collection errors to
        pytest_collectreport hook.
        """
        node_collection_items = list(self.node2collection.items())
        first_node, col = node_collection_items[0]
        same_collection = True
        for node, collection in node_collection_items[1:]:
            msg = report_collection_diff(
                col, collection, first_node.gateway.id, node.gateway.id
            )
            if msg:
                same_collection = False
                self.log(msg)
                if self.config is not None:
                    rep = CollectReport(
                        node.gateway.id, "failed", longrepr=msg, result=[]
                    )
                    self.config.hook.pytest_collectreport(report=rep)

        return same_collection
