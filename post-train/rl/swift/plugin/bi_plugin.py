import re
import textwrap
import pandas as pd
import json
import traceback
import os
import shutil
import sqlite3
import glob
import csv
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List
from datetime import datetime

import torch

from swift.llm import PtEngine, RequestConfig, Template, to_device
from swift.utils import get_logger
from swift.plugin import rm_plugins

from bi_lib.sql_tools_gt import execute_tool as execute_sql_tool
from bi_lib.prompt import PROMPT_SQL_TOOL, TOOLS_SQL, DEFAULT_SYSTEM_PROMPT, FUNCTION_STOPPER
from bi_lib.bi_utils import (
    create_temp_folder,
    compare_with_timeout,
    sanitize_name,
    parse_tool_call_string_sql
)

if TYPE_CHECKING:
    from swift.llm.infer.protocol import ChatCompletionResponse

logger = get_logger()


class DefaultRMPlugin:
    """
    Default Reward Model Plugin

    This class implements the default processing logic for reward models.
    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class GenRMPlugin(DefaultRMPlugin):

    def __init__(self, model, template):
        """
        Generative Reward Model Plugin Example.

        This method sets up the reward model plugin by initializing the PtEngine for efficient inference,
        configuring the request parameters, and defining the system prompt that guides the reward model in
        evaluating responses.

        Args:
            model (torch.nn.Module): The generative reward model.
            template (Template): The template used for encoding input data.
    """

        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig()  # customise your request config here
        self.system = textwrap.dedent("""
            Based on the dialogue history, analyze in detail whether the model's response is accurate, complete, and relevant.
            Assign a reward score between 0 and 1, where 0 indicates completely incorrect and 1 indicates fully correct.
            Before finishing your response, please assign a reward using the following format:

            Reward: {reward}

            For example:
            Reward: 0.85
        """)  # noqa

    def __call__(self, inputs, **kwargs):
        """
        Compute reward scores for the provided inputs.

        This method processes each input by converting dialogue messages into a query, sending the query to the
        reward model for inference, and extracting the reward scores from the model's responses. The final reward
        for each input is the average of all extracted scores.
        Args:
            inputs (List[Dict]): A list of input requests. Each input request is a dictionary containing:
                - 'messages' (List[Dict]): messages from the training model. Each message dictionary includes:
                    - 'role' (str): The role of the speaker (e.g., 'user', 'assistant').
                    - 'content' (str): The content of the message.
                - Additional dataset columns as key-value pairs (e.g., 'solutions', 'images').
        Returns:
            torch.Tensor: A tensor containing the average reward scores for each input. The tensor has a shape of (N,),
            where N is the number of input requests.
        """
        folder = inputs[0]["messages"][0]["content"]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        temp_folder = create_temp_folder(folder)
        with open("/datadrive/chuxuan/ms-swift/rft_data/training_cases.json", "r") as f:
            queries = json.load(f)
        query = queries[folder]["query"]

        csv_list = [f for f in glob.glob(f"{temp_folder}/*.csv") if not os.path.basename(f).startswith("_")]
        df_list, data_list = {}, []
        for _csv in csv_list:
            try:
                table_name = sanitize_name(os.path.basename(_csv)[:-4])
                df = pd.read_csv(_csv, low_memory=False, dtype=str, quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
                df.columns = [sanitize_name(col) for col in df.columns]
                df_list[table_name] = df
                data_list.append(table_name)
            except:
                pass
        try:
            db_path = f"/datadrive/chuxuan/ms-swift/swift/temp/sqlite_db/sqlite_db_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.db"
            conn = sqlite3.connect(db_path)
            for table_name, df in df_list.items():
                df.to_sql(table_name, conn, if_exists='replace', index=False)
        except:
            pass

        prompt = PROMPT_SQL_TOOL.format(query=query, all_files=data_list)
        system_prompt = DEFAULT_SYSTEM_PROMPT + str(TOOLS_SQL + FUNCTION_STOPPER)
        input_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        max_retries, i = 10, 0
        tool_call_seq = []
        state, result = {}, None

        while i < max_retries:
            i += 1

            engine_output = self.engine.infer(input_messages)
            message = engine_output[0].choices[0].message.content

            print(message)
            tool_calls = parse_tool_call_string_sql(message)

            if not tool_calls:
                break

            input_messages.append({
                "role": "assistant",
                "content": message
            })

            responses = []
            stop = False
            for tool_call in tool_calls:
                if tool_call.function.name == "stopper":
                    stop = True
                    break

                tool_call_seq.append(tool_call.function.name)
                try:
                    result_dict = execute_sql_tool(tool_call, queries, folder, conn, temp_folder, data_list)
                    execution_res = result_dict.get("execution_results", {})
                    output = execution_res.get("outputs", "")
                    if result_dict.get("tool_name", None) == "execute_code":
                        code = execution_res.get("code", "")
                        tmp_res = execution_res.get("result", None)
                        state = execution_res.get("state", state)

                        if isinstance(tmp_res, pd.DataFrame):
                            _tmp_res = tmp_res.head()
                            tmp_res_str = _tmp_res.to_string(index=False)
                        else:
                            tmp_res_str = str(tmp_res)

                        res_result = json.dumps({"result (first 5 rows)": tmp_res_str, "outputs": output})
                        if tmp_res is not None:
                            result = tmp_res
                    else:
                        res_result = json.dumps({"outputs": output})

                except Exception:
                    tb = traceback.format_exc()
                    res_result = json.dumps({
                        "outputs": f"[ERROR] Exception occurred during tool execution:\n{tb}"
                    })

                responses.append(res_result)

            # Append tool result back to message list
            input_messages.append({
                "role": "user",
                "content": f"The responses of {message} is {str(responses)}"
            })

            if stop:
                break

        conn.close()
        if os.path.exists(db_path):
            os.remove(db_path)

        # Evaluate result vs ground truth
        if result is not None:
            if isinstance(result, (int, float, str)):
                result = pd.DataFrame({"result": [result]})
            pattern = f"/datadrive/chuxuan/ms-swift/rft_data/gt/{folder}_*.csv"
            file_paths = glob.glob(pattern)
            gt = [pd.read_csv(f) for f in (file_paths or [f"/datadrive/chuxuan/ms-swift/rft_data/gt/{folder}.csv"])]
            try:
                final_res = compare_with_timeout(gt, result)
            except:
                final_res = False
        else:
            final_res = False

        rewards = [1.] if final_res else [0.]
        return torch.tensor(rewards, dtype=torch.float32)

rm_plugins['bi_reward_function'] = GenRMPlugin
print("✅ bi_reward_function registered successfully.")


# prms = {
#     'bi_reward_function': GenRMPlugin
# }