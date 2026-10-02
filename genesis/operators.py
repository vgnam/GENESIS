"""LLM-based heuristic generation operators and the AGRE refinement pipeline.

Operators (paper name in brackets):
    i1      initial generation
    e1      diversity-driven generation           [E1]
    e2      idea-guided recombination              [E2]
    m1      structural modification                [M1]
    m2      parameter variation                    [M2]
    s1      synthesis of several heuristics        [S1]
    counter weakness-exploiting novelty search     [NS]
    refine  Assumption-Guided Reflective Expansion [AGRE]
"""
import logging
import re


class Evolution:

    def __init__(self, model_LLM, prompts, llm_client, agre_stages=None):
        # AGRE stage switches for ablation (missing key = stage enabled)
        self.agre_stages = agre_stages or {}

        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)

        self.model_LLM = model_LLM
        self.interface_llm = llm_client

    # ------------------------------------------------------------------ prompts

    def get_prompt_post_thought(self, code, algorithm):
        prompt_content = self.prompt_task + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    def get_prompt_i1(self):
        prompt_content = self.prompt_task + "\n"
        prompt_content += "First, describe the design idea and main steps of your algorithm in one sentence. " + "The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n\n" \
                         + prompt_indiv
        prompt_content += "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value. \n" \
                         "First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv
        prompt_content += f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n" \
                         f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_m1(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n"
        prompt_content += "Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n"
        prompt_content += "Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_s1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv
        prompt_content += f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n" \
                         "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content

    def get_prompt_counter(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows.\n\n" \
                                            "Algorithm's description: " + indiv1['algorithm'] + "\n" \
                                                                                                "Code:\n" + indiv1[
                             'code'] + "\n"
        prompt_content += "Please analyze the provided algorithm carefully to identify any weaknesses, inefficiencies, or limitations in its design or implementation.\n" \
                                       "Then, create a new algorithm that specifically exploits these weaknesses to outperform or counter the original one.\n" \
                                       "Focus on areas where the opponent's approach is suboptimal or vulnerable, and redesign or optimize those parts.\n" \
                                       "First, describe the design idea based on the provided algorithm and the main steps of the new algorithm in one sentence. " \
                                       "The description must be inside a brace outside the code implementation. " \
                                       "Next, implement it in Python as a function named '" + self.prompt_func_name + "'.\n" \
                                                                                                                      "This function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " + self.joined_inputs + ". " \
                                                                                 "The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " + self.joined_outputs + ". " \
                         + self.prompt_inout_inf + " " + self.prompt_other_inf + "\n" \
                                                                                 "Do not give additional explanations. Use only standard English letters and numbers. Do not output any special characters."
        return prompt_content
    def get_prompt_error_signal(self, indiv):
        prompt = (
                self.prompt_task + "\n\n"
                                   "You are given an algorithm and its implementation.\n\n"
                                   "Algorithm description:\n"
                + indiv['algorithm'] + "\n\n"
                                       "Code:\n"
                + indiv['code'] + "\n\n"
                                  "Your task is NOT to judge correctness.\n"
                                  "Your task is to surface *subtle discomforts* or *uneasy feelings* about this algorithm.\n\n"
                                  "Think like a human expert who says:\n"
                                  "\"I can't prove it's wrong yet, but something here feels fragile or over-assumed.\"\n\n"
                                  "Focus on:\n"
                                  "- Implicit assumptions that may not always hold\n"
                                  "- Design choices that feel brittle or overly confident\n"
                                  "- Parts that might fail silently rather than obviously\n\n"
                                  "Strict rules:\n"
                                  "- Do NOT propose solutions\n"
                                  "- Do NOT suggest fixes\n"
                                  "- Do NOT rewrite code\n"
                                  "- Do NOT explain in detail\n\n"
                                  "List 3–5 brief uncertainty signals (one line each):\n"
                                  "- "
        )
        return prompt

    def get_prompt_counterfactual(self, indiv, error_signal):
        prompt = (
                self.prompt_task + "\n\n"
                                   "You are given an algorithm and its implementation.\n\n"
                                   "Algorithm description:\n"
                + indiv['algorithm'] + "\n\n"
                                       "Code:\n"
                + indiv['code'] + "\n\n"
                                  "Previously identified uncertainty signals:\n"
                + error_signal + "\n\n"
                                 "Now assume the algorithm's behavior is WRONG.\n\n"
                                 "Your goal is to imagine the *simplest possible situation* where this algorithm fails.\n"
                                 "The failure should come from assumptions breaking, not from extreme or unrealistic cases.\n\n"
                                 "Strict rules:\n"
                                 "- Do NOT defend the algorithm\n"
                                 "- Do NOT propose fixes\n"
                                 "- Do NOT modify the algorithm\n\n"
                                 "Describe ONE minimal failure case:\n"
                                 "- Scenario: (brief, concrete)\n"
                                 "- Why this breaks the algorithm: (conceptual reason)\n"
        )
        return prompt

    def get_prompt_role_conflict(self, indiv, counterfactual):
        prompt = (
                self.prompt_task + "\n\n"
                                   "You are given an algorithm and its implementation.\n\n"
                                   "Algorithm description:\n"
                + indiv['algorithm'] + "\n\n"
                                       "Code:\n"
                + indiv['code'] + "\n\n"
                                  "Identified failure scenario:\n"
                + counterfactual + "\n\n"
                                   "You will analyze this algorithm from three strictly separated roles.\n\n"
                                   "ROLE 1 — Defender:\n"
                                   "Argue why the original design choices are reasonable under the intended assumptions.\n"
                                   "Do NOT address the counterfactual directly.\n\n"
                                   "ROLE 2 — Attacker:\n"
                                   "Argue why the counterfactual exposes a fundamental weakness.\n"
                                   "Do NOT suggest solutions.\n\n"
                                   "ROLE 3 — Observer:\n"
                                   "Summarize the *core unresolved tension* between Defender and Attacker.\n"
                                   "Do NOT choose a winner.\n"
                                   "Do NOT propose fixes.\n\n"
                                   "Output:\n"
                                   "Defender:\n"
                                   "- ...\n\n"
                                   "Attacker:\n"
                                   "- ...\n\n"
                                   "Observer (conflicts only):\n"
                                   "- ...\n"
        )
        return prompt

    def get_prompt_abstraction(self, role_conflict):
        prompt = (
                self.prompt_task + "\n\n"
                                   "Below is a multi-perspective analysis of an algorithm.\n\n"
                                   "Analysis:\n"
                + role_conflict + "\n\n"
                                  "Your task is to step back and reason at a higher level of abstraction.\n\n"
                                  "Ignore all implementation details.\n"
                                  "Think in terms of *strategy*, *assumptions*, and *trade-offs*.\n\n"
                                  "Output format:\n"
                                  "Core strategy (1–2 bullets):\n"
                                  "- ...\n\n"
                                  "Hidden or risky assumptions:\n"
                                  "- ...\n"
                                  "- ...\n"
        )
        return prompt

    def get_prompt_assumption_repair(self, abstraction):
        prompt = (
                self.prompt_task + "\n\n"
                                   "Below are abstract principles and risky assumptions of an algorithm.\n\n"
                + abstraction + "\n\n"
                                "Your task is to revise ONLY the assumptions or applicability conditions.\n\n"
                                "You are NOT allowed to:\n"
                                "- Design a new algorithm\n"
                                "- Suggest new mechanisms\n"
                                "- Change the overall strategy\n\n"
                                "Only clarify, restrict, or condition the assumptions to make failures less likely.\n\n"
                                "Revised assumptions:\n"
                                "- ...\n"
                                "- ...\n"
        )
        return prompt

    def get_prompt_final_advice(self, repaired_assumptions):
        prompt = (
            self.prompt_task + "\n\n"
            "You are given revised assumptions that improve the robustness of an algorithm.\n\n"
            "Revised assumptions:\n"
            + repaired_assumptions + "\n\n"
            "Based ONLY on these assumptions, provide actionable guidance for improving the algorithm.\n\n"
            "Output in the following structured format:\n\n"
            "Improvement Direction: [short, concrete label]\n"
            "Rationale: [why adjusting the algorithm under these assumptions helps]\n"
            "Implementation Guidance:\n"
            "- ...\n"
            "- ...\n"
            "Expected Impact: [robustness / stability / generalization gain]\n\n"
            "Cautions / What to Avoid:\n"
            "- ...\n"
            "- ...\n"
        )
        return prompt
    def get_prompt_refine_with_critic(self, indiv, advice):

        prompt_content = self.prompt_task + "\n"

        prompt_content += (
                "You are an expert algorithm engineer and software developer with extensive experience in refining "
                "heuristic and optimization algorithms for reliability and performance. "
                "You are given a heuristic algorithm and its current Python implementation.\n\n"

                "Current code:\n"
                + indiv['code'] + "\n\n"
                         "A Critic has analyzed this implementation and produced the following feedback:\n"
                + advice + "\n\n"
                                    "Your task is to performe a refinement step by carefully revising "
                                    "the existing implementation to address the weaknesses identified by the Critic, while preserving "
                                    "the parts of the logic that already work well. "
                                    "You must improve robustness, correctness, clarity, and efficiency without changing the core "
                                    "algorithmic idea, the intended behavior, or the input–output interface.\n\n"

                                    "First, briefly describe the refined algorithm’s design idea and main steps in one sentence; "
                                    "this description must be enclosed in braces and placed outside the code implementation. "
                                    "Then, implement the refined version in Python as a function named '"
                + self.prompt_func_name + "'. "
                                          "The function should accept "
                + str(len(self.prompt_func_inputs)) + " input(s): "
                + self.joined_inputs + ". "
                + self.prompt_inout_inf + " "
                + self.prompt_other_inf + " "
                                          "Do not provide any additional explanations outside the required description and code. "
                                          "Ensure all necessary imports are included and that the code has no syntax errors."
        )

        return prompt_content

    # ------------------------------------------------------------------ LLM calls

    def _get_thought(self, prompt_content, step=""):
        logging.debug(f">>> prompt for [ {step} ]:\n{prompt_content}")
        response = self.interface_llm.get_response(prompt_content)
        logging.debug(f">>> response of [ {step} ]:\n{response}")
        return response

    @staticmethod
    def _parse_response(response):
        """Split a response into (description inside braces, code from the first import/def to the last return)."""
        algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)
        return algorithm, code

    def _get_alg(self, prompt_content, step=""):
        algorithm, code = self._parse_response(self._get_thought(prompt_content, step))

        n_retry = 1
        while len(algorithm) == 0 or len(code) == 0:
            logging.debug("Algorithm or code not identified, retrying ...")
            algorithm, code = self._parse_response(self._get_thought(prompt_content, step))
            if n_retry > 3:
                break
            n_retry += 1

        # The regex stops at "return"; re-append the returned variable names
        code_all = code[0] + " " + ", ".join(s for s in self.prompt_func_outputs)
        return [code_all, algorithm]

    def post_thought(self, code, algorithm):
        """Ask the LLM to re-describe the generated code (the stored description)."""
        return self._get_thought(self.get_prompt_post_thought(code, algorithm), "post_thought")

    def i1(self):
        return self._get_alg(self.get_prompt_i1(), "i1")

    def e1(self, parents):
        return self._get_alg(self.get_prompt_e1(parents), "e1")

    def e2(self, parents):
        return self._get_alg(self.get_prompt_e2(parents), "e2")

    def m1(self, parent):
        return self._get_alg(self.get_prompt_m1(parent), "m1")

    def m2(self, parent):
        return self._get_alg(self.get_prompt_m2(parent), "m2")

    def s1(self, parents):
        return self._get_alg(self.get_prompt_s1(parents), "s1")

    def counter(self, parent):
        return self._get_alg(self.get_prompt_counter(parent), "counter")

    # ------------------------------------------------------------------ AGRE

    def agre(self, indiv):
        """
        Assumption-Guided Reflective Expansion: error signal -> counterfactual failure ->
        role-based conflict -> strategic abstraction -> assumption repair -> final guidance.
        The four middle stages can be disabled via ``agre_stages`` for the ablation study;
        a disabled stage passes its input through to the next one.
        """
        stages = self.agre_stages

        x = self._get_thought(self.get_prompt_error_signal(indiv), "error_signal")
        if stages.get('counterfactual', True):
            x = self._get_thought(self.get_prompt_counterfactual(indiv, x), "counterfactual")
        if stages.get('role_conflict', True):
            x = self._get_thought(self.get_prompt_role_conflict(indiv, x), "role_conflict")
        if stages.get('abstraction', True):
            x = self._get_thought(self.get_prompt_abstraction(x), "abstraction")
        if stages.get('assumption_repair', True):
            x = self._get_thought(self.get_prompt_assumption_repair(x), "assumption_repair")
        return self._get_thought(self.get_prompt_final_advice(x), "final_advice")

    def refine(self, indiv):
        advice = self.agre(indiv)
        return self._get_alg(self.get_prompt_refine_with_critic(indiv, advice), "refine")
