import json
from Task_Generation_sft_batch2_copy.Factories.llm_factory import LLM_factory
class Reward:
    def __init__(self):
        self.llm = LLM_factory()
        return
    def turn_level_reward(self, generated_trajectory, ground_truth_messages):
        """
        Reward each turn (tool call, reasoning step) in the trajectory
        """
        rewards = []
        
        # for turn_idx, msg in enumerate(generated_trajectory):
        #     if msg['role'] == 'assistant':
        #         reward = 0.0
                
        #         # Tool execution reward (0.2 if tool correctly called)
        #         if 'tool_calls' in msg:
        #             gt_tool = ground_truth_messages[turn_idx].get('tool_calls')
        #             if gt_tool and msg['tool_calls'][0]['function']['name'] == gt_tool[0]['function']['name']:
        #                 reward += 0.2
                
        #         # Tool arguments correctness (0.3 if args match)
        #         if 'tool_calls' in msg:
        #             gt_args = ground_truth_messages[turn_idx]['tool_calls'][0]['function']['arguments']
        #             gen_args = msg['tool_calls'][0]['function']['arguments']
        #             if json.loads(gt_args) == json.loads(gen_args):
        #                 reward += 0.3
                
        #         rewards.append(reward)
        
        # return sum(rewards) / len(rewards) if rewards else 0.0
        return 0.5
    
    def check_correctness_with_llm(self, query, generated_answer, reference_answer):
        """
        Use LLM to evaluate correctness using the provided Prometheus prompt template
        """
        # print("Generated Answer", generated_answer)
        prompt = f"""###Task Description:
    An instruction (might include an Input inside it), a query, a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
    1. Write a detailed feedback that assesses the quality of the response strictly based on the given score rubric, not evaluating in general.
    2. After writing a feedback, write a score that is either 1 or 2 or 3 or 4 or 5. You should refer to the score rubric.
    3. The output format should only look as follows: "Feedback: (write a feedback for criteria) [an integer number between 1 and 5] "
    4. Please do not generate any other opening, closing, and explanations.
    5. Only evaluate on common things between generated answer and reference answer. Don't evaluate on things which are present in reference answer but not in generated answer.

    ###Instruction:
    Your task is to evaluate the generated answer and reference answer for the following query:
    {query}


    ###Generate answer to evaluate:
    {generated_answer}


    ###Reference Answer (Score 5):
    {reference_answer}


    ###Score Rubrics:
    Score 1: If the generated answer is not relevant to the user query and reference answer with missing entities.
    Score 2: If the generated answer is relevant to user query or contains significant mistakes or misses important entities or if entities present in reference answer are not exact match.
    Score 3: If the generated answer is relevant to the reference answer and answers the query but contains very few mistakes.
    Score 4: If the generated answer is relevant to the user query and all the entities are exact match as the reference answer along with same intent, but it is not as concise.
    Score 5: If the generated answer is relevant to the user query and fully correct according to the reference answer.


    ###Feedback:"""

        response = self.llm.gpt(prompt)
        
        output = response.content
        
        try:
            # Extract score from "Feedback: ... [X]" format
            score_match = output.split('[')[-1].split(']')[0].strip()
            score = int(score_match)
            # Normalize to 0-1 scale: (score - 1) / 4
            normalized_score = (score - 1) / 4.0
            return normalized_score
        except:
            return 0.0


    def check_relevancy_with_llm(self, query_str, context_str, existing_answer=""):
        """
        Use LLM to evaluate relevancy using the provided Prometheus relevancy prompt template
        """
        prometheus_relevancy_refine_prompt_template = f"""###Task Description:
    An instruction (might include an Input inside it), a query with response, context, an existing answer, and a score rubric representing a evaluation criteria are given.
    1. You are provided with evaluation task with the help of a query with response and context and an existing answer.
    2. Write a detailed feedback based on evaluation task and the given score rubric, not evaluating in general.
    3. After writing a feedback, write a score that is YES or NO. You should refer to the score rubric.
    4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (YES or NO)"
    5. Please do not generate any other opening, closing, and explanations.


    ###The instruction to evaluate: Your task is to evaluate if the response for the query is in line with the context information provided.


    ###Query and Response:
    {query_str}


    ###Context:
    {context_str}


    ###Score Rubrics:
    Score YES: If the existing answer is already YES or If the response for the query is in line with the context information provided.
    Score NO: If the existing answer is NO and If the response for the query is in line with the context information provided.


    ###Feedback: """

        response = self.llm.gpt(prometheus_relevancy_refine_prompt_template)
        
        output = response.content
        
        try:
            # Extract YES/NO from "[RESULT] (YES or NO)" format
            if 'YES' in output.upper():
                return 1.0
            elif 'NO' in output.upper():
                return 0.0
            else:
                return 0.0
        except:
            return 0.0


    def outcome_level_reward(self, generated_trajectory, ground_truth_messages, original_query):
        """
        Reward the final answer quality based on Correctness and Relevancy
        """
        # Get final assistant message
        final_gen_list = [m for m in generated_trajectory if m.get('role') == 'assistant']
        final_gt_list = [m for m in ground_truth_messages if m.get('role') == 'assistant']
        
        if not final_gen_list or not final_gt_list:
            return 0.0
        
        final_gen = final_gen_list[-1]
        final_gt = final_gt_list[-1]
        
        gen_content = final_gen.get('content', '')
        gt_content = final_gt.get('content', '')
        
        # Evaluate Correctness (1-5 scale normalized to 0-1)
        correctness_score = self.check_correctness_with_llm(
            query=original_query,
            generated_answer=generated_trajectory,
            reference_answer=ground_truth_messages
        )
        
        # Evaluate Relevancy (YES/NO converted to 1.0/0.0)
        # Context is the reference answer for relevancy check
        query_and_response = f"Query: {original_query}\nResponse: {ground_truth_messages}"
        relevancy_score = self.check_relevancy_with_llm(
            query_str=query_and_response,
            context_str=generated_trajectory
        )
        
        # Combined reward: 50% correctness + 50% relevancy
        total_reward = 0.5 * correctness_score + 0.5 * relevancy_score
        
        return total_reward

    def compute_reward(self, prompt, generated_trajectory, dataset_item):
        """
        Main reward function for GRPO trainer
        """
        # ground_truth = dataset_item['messages']
    
        # # Extract user query from prompt
        # user_query = [msg['content'] for msg in prompt if msg['role'] == 'user'][0]
        
        # # Compute turn-level reward
        # turn_reward = self.turn_level_reward(generated_trajectory, ground_truth)
        
        # # Compute outcome-level reward with correctness and relevancy
        # outcome_reward = self.outcome_level_reward(
        #     generated_trajectory, 
        #     ground_truth,
        #     user_query  # Pass the original query
        # )
        
        # # Combine: 40% turn-level + 60% outcome-level
        # total_reward = 0.4 * turn_reward + 0.6 * outcome_reward
        
        # return total_reward
        return 0.5

