from llmfuzzer.utils.template import synthesis_message


class PromptNode:
    def __init__(self,
                 fuzzer,
                 prompt,
                 response = None,
                 results = None,
                 parent = None,
                 mutator = None):
        self.fuzzer = fuzzer
        self.prompt = prompt
        self.response = response
        self.results = results
        self.visited_num = 0

        self.parent = parent
        self.mutator = mutator
        self.child = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return self.num_query - self.num_jailbreak

    @property
    def num_query(self):
        return len(self.results)


class LLMFuzzer:
    def __init__(self,
                 question,
                 target_model,
                 predictor,
                 initial_seed,
                 mutate_policy,
                 select_policy,
                 logger,
                 energy,
                 max_query,
                 max_jailbreak,
                 max_reject=-1,
                 max_iteration=-1,
                 ):

        self.question = question
        self.target_model = target_model
        self.predictor = predictor
        self.prompt_nodes = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()

        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy

        self.current_query = 0
        self.current_jailbreak = 0
        self.current_reject = 0
        self.current_iteration = 0

        self.max_query = max_query
        self.max_jailbreak = max_jailbreak
        self.max_reject = max_reject
        self.max_iteration = max_iteration

        self.energy = energy
        self.logger = logger

        self.succeeded_prompt = None
        self.succeeded_response = None

        self.setup()

    def setup(self):
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self

    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

    def run(self):
        self.logger.info("Fuzzing started!")
        
        while not self.is_stop():
            seed = self.select_policy.select()
            mutated_results = self.mutate_policy.mutate_single(seed)
            self.evaluate(mutated_results)
            self.update(mutated_results)
            self.log()

        self.logger.info("Fuzzing finished!")
        return {
            "is_succeeded": self.current_jailbreak > 0,
            "num_queries": self.current_query,
            "succeeded_prompt": self.succeeded_prompt,
            "succeeded_response": self.succeeded_response,
        }

    def evaluate(self, prompt_nodes):
        for prompt_node in prompt_nodes:
            message = synthesis_message(self.question, prompt_node.prompt)
            if message is None:
                prompt_node.response = []
                prompt_node.results = []
                break
            response = self.target_model.generate(message)
            prompt_node.response = response
            prompt_node.results = self.predictor.predict([response])

    def update(self, prompt_nodes):
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                self.succeeded_prompt = prompt_node.prompt
                self.succeeded_response = prompt_node.response
                self.logger.info(f"Question: \n{self.question}")
                self.logger.info(f"Prompt: \n{prompt_node.prompt}")
                self.logger.info(f"Response: \n{prompt_node.response}")

            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

        self.select_policy.update(prompt_nodes)

    def log(self):
        self.logger.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, {self.current_reject} rejects, {self.current_query} queries")