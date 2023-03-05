import os
import numpy as np
import pandas as pd
import seaborn as sns

def sort_key(e):
    return e['run']

def generate_table(data):
    all_means = []
    all_stds = []
    combined = []
    
    all_dfs = data['all_dfs']
    
    for i in range(len(all_dfs)): 
        #data = all_dfs[i]['accumulated_reward']
        #print(data, len(data))
        n_rows =  len(all_dfs[i]['accumulated_reward'])
        data = all_dfs[i]['accumulated_reward'][(int(n_rows*0.8)):]
        #print(data, n_rows)

        #print(f"{all_dfs[i]['accumulated_reward'].mean()} +/- {all_dfs[i]['accumulated_reward'].std()}")
        all_means.append(data.mean())
        all_stds.append(data.std())
        combined.append( "{:5.4f}".format(all_means[-1]) + " \u00B1 " + "{:3.4f}".format(all_stds[-1]) )
        #print("{:5.4f}".format(all_means[-1]), " \u00B1 ", "{:3.4f}".format(all_stds[-1]))

    
    #print()
    #print("\033[1m{:5.4f}".format(np.mean(all_means)), " \u00B1 ", "{:3.4f}".format(np.mean(all_stds)), "\033[0m")
    #print()
    all_means.append(np.mean(all_means))
    all_stds.append(np.mean(all_stds))
    combined.append( "{:5.4f}".format(np.mean(all_means)) + " \u00B1 " + "{:3.4f}".format(np.mean(all_stds)) )

    
    return all_means, all_stds, combined

def generate_plot(data, plot: str = 'reward', max_steps: int = 200, label=None, scale='M'):
    all_dfs = data['all_dfs']
    
    # 4. trim each dataset so that all dataset have an equal amount of episodes  
    length = 1_000_000_000_000
    for df in all_dfs:
        if len(df) < length:
            length = len(df)

    for i in range(len(all_dfs)):
        all_dfs[i] = all_dfs[i].truncate(after=length-1)
            
    # 4. create a new data frame with all datasets appended
    for i in range(len(all_dfs)):
        if i == 0:
            df = all_dfs[i]
        else:
            df = df.append(all_dfs[i], ignore_index=True)

    #print(all_dfs)
    #print(df)
    df = df.assign(episode_timesteps=df["episode_number"]*max_steps)
    # 5. plot a line plot using seaborn and is should show a 95% ci.        
    #sns.lineplot(data=df, x='episode_number', y='accumulated_reward')
    #sns.lineplot(data=df, x='episode_number', y='accumulated_reward_sma')
    #print(j, args.labels[j])
    #if args.window is not None:
    #    sns.lineplot(data=df, ci="sd", x='episode_number', y='accumulated_reward_sma', palette=palletes[j], label=args.labels[j])
    #else:
    #    sns.lineplot(data=df, ci="sd", x='episode_number', y='accumulated_reward', palette=palletes[j], label=args.labels[j])
    #label = f"{data['batch'][0]['experiment']}:{data['batch'][0]['algorithm']}"
    if label == None:
        label = f"{data['batch'][0]['algorithm']}"
    
    if plot == 'reward':
        f = sns.lineplot(data=df, ci="sd", x='episode_timesteps', y='accumulated_reward',label=label)
    elif plot == 'episode_length':
        f = sns.lineplot(data=df, ci="sd", x='episode_timesteps', y='episode_length',label=label)
    else:
        print(f'plot={plot} is not a supported option.')
    if scale == 'M':
        xlabels = ['{:,.2f}'.format(x) + 'M' for x in f.get_xticks()/1000_000]
    else:
        xlabels = ['{:}'.format(int(x)) + 'k' for x in f.get_xticks()/1_000]

    f.set_xticklabels(xlabels)
    
def generate_multi_agent_plot(batch, window: int = -1, plot: str = 'reward', max_steps: int = 200, label=None, scale='k'):

    for b in batch:
        print(b)
        full_path = os.path.join(b['dir'], 'training.log')

        f = open(full_path, 'rb')   # open file in read mode and binary

        episode_num = []
        timesteps = []
        timesteps_ax = []
        episode_length = []
        episode_reward = []
        read_data = False

        for line in f:
            l = str(line)[2:-5]
            if l == "data begin":
                #print('Found tag: \'data begin\'')
                read_data = True
                continue

            if l == "data end":
                #print('Found tag: \'data end\'')
                read_data = False
                continue

            if read_data:
                fields = l.split(',')
                if fields[0] == "episode_number":
                    #print(f'header: {l}')
                    hdr = l.split(',')
                    continue
                episode_num.append(int(fields[0]))
                timesteps.append(int(fields[1]))        # total timesteps to date
                #if len(timesteps_ax) != 0:
                    #timesteps_ax.append(timesteps_ax[-1] + timesteps[-1])
                #else:
                    #timesteps_ax.append(0)
                episode_length.append(int(fields[2]))
                episode_reward.append(np.float32(fields[3]))
            #    print(l, fields)

        f.close()
        
        #fig, ax = plt.subplots()
        #ax.plot(timesteps_ax, episode_reward)
        #plt.savefig(f'SAD_log_fit_{str(step_size).replace(".","_")}mm.png')
        df = pd.DataFrame({'t': timesteps, 'episode_return':episode_reward} )
        if window > 0:
            avg = df[['episode_return']].rolling(center=False, window=int(window), min_periods=1).mean()
            df=df.assign(episode_return=avg["episode_return"]) # append column to the end of the dataframe.
        
        f = sns.lineplot(data=df,
                         x='t',
                         y='episode_return',
                         label=f'run_{b["run"]}' if label==None else label+f' - run_{b["run"]}')
        
    if scale == 'M':
        xlabels = ['{:,.2f}'.format(x) + 'M' for x in f.get_xticks()/1000_000]
    else:
        xlabels = ['{:}'.format(int(x)) + 'k' for x in f.get_xticks()/1_000]

    f.set_xticklabels(xlabels)

def generate_multi_agent_plot_w_mean_std(batch, window: int = -1, plot: str = 'reward', max_steps: int = 200, label=None, scale='k'):
    all_dfs = []
    for b in batch:
        print(b)
        full_path = os.path.join(b['dir'], 'training.log')

        f = open(full_path, 'rb')   # open file in read mode and binary

        episode_num = []
        timesteps = []
        timesteps_ax = []
        episode_length = []
        episode_reward = []
        read_data = False

        for line in f:
            l = str(line)[2:-5]
            if l == "data begin":
                #print('Found tag: \'data begin\'')
                read_data = True
                continue

            if l == "data end":
                #print('Found tag: \'data end\'')
                read_data = False
                continue

            if read_data:
                fields = l.split(',')
                if fields[0] == "episode_number":
                    #print(f'header: {l}')
                    hdr = l.split(',')
                    continue
                episode_num.append(int(fields[0]))
                timesteps.append(int(fields[1]))        # total timesteps to date
                #if len(timesteps_ax) != 0:
                    #timesteps_ax.append(timesteps_ax[-1] + timesteps[-1])
                #else:
                    #timesteps_ax.append(0)
                episode_length.append(int(fields[2]))
                episode_reward.append(np.float32(fields[3]))
            #    print(l, fields)

        f.close()
        
        #fig, ax = plt.subplots()
        #ax.plot(timesteps_ax, episode_reward)
        #plt.savefig(f'SAD_log_fit_{str(step_size).replace(".","_")}mm.png')
        df = pd.DataFrame({'t': timesteps, 'episode_return':episode_reward, 'episode_number':episode_num} )
        if window > 0:
            avg = df[['episode_return']].rolling(center=False, window=int(window), min_periods=1).mean()
            df=df.assign(episode_return=avg["episode_return"]) # append column to the end of the dataframe.
        
        all_dfs.append(df)

    # 4. trim each dataset so that all dataset have an equal amount of episodes  
    length = 1_000_000_000_000
    for df in all_dfs:
        if len(df) < length:
            length = len(df)

    for i in range(len(all_dfs)):
        all_dfs[i] = all_dfs[i].truncate(after=length-1)
        
    # 4. create a new data frame with all datasets appended
    for i in range(len(all_dfs)):
        if i == 0:
            df = all_dfs[i]
        else:
            df = df.append(all_dfs[i], ignore_index=True)
        #f = sns.lineplot(data=df,
                         #x='t',
                         #y='episode_return',
                         #label=f'run_{b["run"]}' if label==None else label+f' - run_{b["run"]}')
    #df = df.assign(episode_timesteps=df["episode_number"]*max_steps)             
    
    f = sns.lineplot(data=df, ci='sd', x='episode_number', y='episode_return',label=label)
        
    if scale == 'M':
        xlabels = ['{:,.2f}'.format(x) + 'M' for x in (f.get_xticks()*200)/1000_000]
    else:
        xlabels = ['{:}'.format(int(x)) + 'k' for x in (f.get_xticks()*200)/1_000]

    f.set_xticklabels(xlabels)

def generate_dataset(batch, window: int = -1):
    batch.sort(key=sort_key)    # sort batch in ascending run numbers, so that automatically generated table row descriptors correspond to the proper run numbers
    
    data = {'batch': batch}
    all_dfs = []
    for b in batch:
        #print(b['experiment'], b['algorithm'])
        full_path = os.path.join(b['dir'], 'training.log')

        f = open(full_path, 'rb')   # open file in read mode and binary

        episode_num = []
        timesteps = []
        episode_length = []
        episode_reward = []
        read_data = False

        for line in f:
            l = str(line)[2:-5]
            if l == "data begin":
                #print('Found tag: \'data begin\'')
                read_data = True
                continue

            if l == "data end":
                #print('Found tag: \'data end\'')
                read_data = False
                continue

            if read_data:
                fields = l.split(',')
                if fields[0] == "episode_number":
                    #print(f'header: {l}')
                    hdr = l.split(',')
                    continue
                episode_num.append(int(fields[0]))
                timesteps.append(int(fields[1]))
                episode_length.append(int(fields[2]))
                episode_reward.append(np.float32(fields[3]))
            #    print(l, fields)

        f.close()

        episode_num = np.array(episode_num, dtype=np.int32)
        timesteps = np.array(timesteps, dtype=np.int32)
        episode_length = np.array(episode_length, dtype=np.int32)
        episode_reward = np.array(episode_reward, dtype=np.float32)

        df = pd.DataFrame(data = {hdr[0]: episode_num, hdr[1]: timesteps, hdr[2]: episode_length, hdr[3]: episode_reward})
            #print(df)

            # 3. compute a rolling average
            # If you pass min_periods=1 then it will take the first value as it is, second value as the mean of the first two etc. It might make more sense in some cases.
        if window > 0:
            avg = df[['episode_length', 'accumulated_reward']].rolling(center=False, window=int(window), min_periods=1).mean()
            avg.rename({"episode_length": "episode_length_sma", "accumulated_reward": "accumulated_reward_sma"}, inplace=True, axis=1)
            #print(avg)
            df=df.assign(episode_length=avg["episode_length_sma"]) # append column to the end of the dataframe.
            df=df.assign(accumulated_reward=avg["accumulated_reward_sma"])
            #print(df)
                
        all_dfs.append(df)
        
    data['all_dfs'] = all_dfs
    
    return data
