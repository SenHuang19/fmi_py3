from testcase import TestCase
import json

config = './config'

with open(config) as json_file:
        model_config = json.load(json_file)
        
case = TestCase(model_config)

print(case.input_names)

print(case.get_measurements())


case.initialize(start_time=600, warmup_period=60)

y = case.advance({'oveAct_activate':1, 'oveAct_u':0})

print(y)

y = case.advance({'oveAct_activate':1, 'oveAct_u':1})

print(y)