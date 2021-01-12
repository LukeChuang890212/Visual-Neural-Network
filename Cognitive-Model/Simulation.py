#%%
from connections import pd, np

from network import VisualNetWork

#%%
input_data = pd.read_excel("input 2.xlsx")

salience = 100.0

valid_trials = input_data[input_data["Trial"] == "valid"].drop("Trial", axis = 1).to_numpy()*salience
invalid_trials = input_data[input_data["Trial"] == "invalid"].drop("Trial", axis = 1).to_numpy()*salience
neutral_trials = input_data[input_data["Trial"] == "neutral"].drop("Trial", axis = 1).to_numpy()*salience

# print(len(valid_trials))
# print(len(invalid_trials))
# print(len(neutral_trials))

valid_trials = np.array_split(valid_trials,len(valid_trials)/4)
invalid_trials = np.array_split(invalid_trials,len(invalid_trials)/4)
neutral_trials = np.array_split(neutral_trials,len(neutral_trials)/2)

# print(len(valid_trials))
# print(len(invalid_trials))
# print(len(neutral_trials))

#%%
def simulate(target, cue, VisualNetWork, record, not_neutral):
    print("-----------------")
    print("Trial type: {}".format(record[-1][0]),'\n')
    if(not_neutral):
        print("Cue:")
        print(cue,'\n')
    print("Target:")
    print(target,'\n')

    print("Result:",'\n')
    model = VisualNetWork(wt=0.11, cascade_rate=0.8) #0.987985
    if(not_neutral):
        model(cue, iscue = True)
    record[-1] += model(target,iscue = False)
    print("-----------------")

    return record

#%%
record = []
for trial_type in enumerate([valid_trials, invalid_trials, neutral_trials]):
    for trial in [trial_type[1][0]]:
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
# invalid = input_data[input_data["Trial"] == "invalid trial"].drop("Trial", axis = 1)
# all_zero_time = 0
# pattern = []
# for i in range(0,len(invalid),4):
#     for j in range(i,i+4):
#         pattern.append(int(np.array_equal(invalid.iloc[j,:],np.zeros(7))))
#         print(int(np.array_equal(invalid.iloc[j,:],np.zeros(7))))
#     if(int(np.array_equal(a,[1,0,0,1])) == 0):
#         print("i:",i)
#         break
#     else:
#         pattern = []

# invalid.iloc[i:i+4,:]