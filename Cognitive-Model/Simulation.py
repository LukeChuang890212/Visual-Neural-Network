#%%
from connections import pd, np
import matplotlib.pyplot as plt

from network import VisualNetWork

class Sim():
    def __init__(self):
        # 讀input資料
        #%% 
        input_data = pd.read_excel("input_data/input 2.xlsx")

        salience = 100.0

        self.valid_trials = input_data[input_data["Trial"] == "valid"].drop("Trial", axis = 1).to_numpy()*salience
        self.invalid_trials = input_data[input_data["Trial"] == "invalid"].drop("Trial", axis = 1).to_numpy()*salience
        self.neutral_trials = input_data[input_data["Trial"] == "neutral"].drop("Trial", axis = 1).to_numpy()*salience

        # print(len(valid_trials))
        # print(len(invalid_trials))
        # print(len(neutral_trials))

        self.valid_trials = np.array_split(self.valid_trials,len(self.valid_trials)/4)
        self.invalid_trials = np.array_split(self.invalid_trials,len(self.invalid_trials)/4)
        self.neutral_trials = np.array_split(self.neutral_trials,len(self.neutral_trials)/2)

        # print(len(valid_trials))
        # print(len(invalid_trials))
        # print(len(neutral_trials))

    #%%
    def simulate(self, target, cue, VisualNetWork, record, not_neutral):
        print("----------------------------------------------------")
        print("Trial type: {}".format(record[-1][0]),'\n')
        if(not_neutral):
            print("Cue:")
            print(cue,'\n')
        print("Target:")
        print(target,'\n')

        print("Result:",'\n')
        # 調weight(wt)和cascade_rate的地方
        model = VisualNetWork(wt=0.2, cascade_rate=0.9, bias=-0.3) # 正常人 wt=0.11, cascade_rate=0.8, bias=-0.5
        if(not_neutral):
            model(cue, iscue = True)
        record[-1] += model(target,iscue = False)
        print("----------------------------------------------------")

        return record

    def run(self):
        #%%
        record = []
        for trial_type in enumerate([self.valid_trials, self.invalid_trials, self.neutral_trials]):
            for trial in [trial_type[1][0]]:
                if trial_type[0] != 2: 
                    cue, target = np.array_split(trial,2)

                    if trial_type[0] == 0:
                        record.append(["valid"])
                    elif trial_type[0] == 1:
                        record.append(["invalid"])
                    
                    record = self.simulate(target, cue, VisualNetWork, record, not_neutral=True)
                else:
                    target = trial
                    record.append(["neutral"])

                    record = self.simulate(target, cue, VisualNetWork, record, not_neutral=False)
                
        record = pd.DataFrame(record)
        record.columns = ["Trial","Output","Cycle"]
        print("Record")
        print(record)

        plt.plot(record["Trial"],record["Cycle"], marker='o')
        # plt.title('title name')
        plt.xlabel('Trial Type')
        plt.ylabel('Number of Cycle')
        plt.show()    

#%%
def main():
    sim = Sim()
    sim.run()

main()
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