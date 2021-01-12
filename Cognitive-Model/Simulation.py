#%%
from network import VisualNetWork
import pandas as pd 
from connections import np

#%%
input_data = pd.read_excel("input new.xlsx").dropna()

valid_trials = input_data[input_data["Trial"] == "valid trial"].drop("Trial", axis = 1).to_numpy().astype(int)
invalid_trials = input_data[input_data["Trial"] == "invalid trial"].drop("Trial", axis = 1).to_numpy().astype(int)
neutral_trials = input_data[input_data["Trial"] == "neutral trial"].drop("Trial", axis = 1).to_numpy().astype(int)

valid_trials = np.array_split(valid_trials,len(valid_trials)/4)
invalid_trials = np.array_split(invalid_trials,len(invalid_trials)/4)
neutral_trials = np.array_split(neutral_trials,len(neutral_trials)/2)
# print(valid_trials)
# print(invalid_trials)
# print(neutral_trials)

#%%
def simulate(target, cue, VisualNetWork, record, not_neutral):
    print("-----------------")
    print("Trial type: {}".format(record[-1][0]),'\n')
    print("Cue:")
    print(cue,'\n')
    print("Target:")
    print(target,'\n')

    print("Result:",'\n')
    model = VisualNetWork(wt=1, cascade_rate=0.987985)
    if(not_neutral):
        model(cue, iscue = True)
    record[-1] += model(target,iscue = False)
    print("-----------------")

    return record

#%%
record = []
for trial_type in enumerate([valid_trials, invalid_trials, neutral_trials]):
    for trial in trial_type[1]:
        # print(len(trial))
        if len(trial) != 2 and len(trial) != 4:
            continue
        if trial_type[0] != 2:
            cue, target = np.array_split(trial,2)

            if trial_type[0] == 0:
                record.append(["valid"])
            elif trial_type[0] == 1:
                record.append(["invalid"])
            
            record = simulate(target, cue, VisualNetWork, record, not_neutral=True)
        else:
            target = trial
            record.append(["neutral"])

            record = simulate(target, cue, VisualNetWork, record, not_neutral=False)
        
        # print("record")
        # print(pd.DataFrame(record))
        

# %%
