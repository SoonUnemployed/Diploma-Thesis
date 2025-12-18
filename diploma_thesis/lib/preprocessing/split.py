from pathlib import Path
import os
import pandas as pd 
from sklearn.model_selection import train_test_split

def split_data(out_path: Path):
    
    files = [f for f in os.listdir(out_path) if f.endswith(".fif")]
    files = [f.split(".")[0] for f in files]

    df = pd.DataFrame({
        "session": files,
        "split": None
    })

    impostors = [("_").join(f.split("_")[:2]) for f in files]
    impostors = list(set([f for f in impostors if impostors.count(f) == 2]))

    users = list(set(["_".join(f.split("_")[:2]) for f in files]))
    users = [i for i in users if i not in impostors]

    impostor_sessions = [f for f in files if ("_").join(f.split("_")[:2]) in impostors]
    #sessions = [f for f in files if ("_").join(f.split("_")[:2]) in users]

    df["user"] = df["session"].str.split("_", expand = True)[1].astype(int)

    used_idx = df[df["session"].isin(impostor_sessions)].index
    #Remove 2 sessions from user08
    temp = df[df["user"] == 8]
    temp = temp.sample(n = 2, random_state = 19).index
    used_idx = used_idx.append(temp).sort_values()
    temp = df.iloc[temp]
    temp = temp["session"].tolist()

    impostor_sessions.extend(temp)

    not_used_df = df.drop(used_idx)
    selected = not_used_df.groupby("user", group_keys = False).apply(
        lambda x: x.sample(n = 1, random_state = 19),
        include_groups = False
    )
    test_users = selected["session"].tolist()
    used_idx = selected.index.tolist()

    not_used_df = not_used_df.drop(used_idx)
    selected = not_used_df.groupby("user", group_keys = False).apply(
        lambda x: x.sample(n = 1, random_state = 19),
        include_groups = False
    )
    val_users = selected["session"].tolist()
    used_idx = selected.index.tolist()

    not_used_df = not_used_df.drop(used_idx)
    train_users = not_used_df["session"].tolist()


    '''
    impostors = ["user_02", "user_04", "user_05"]
    users = list(set(["_".join(f.split("_")[:2]) for f in files]))
    users = [i for i in users if i not in impostors]

    impostor_sessions = [f for f in files if ("_").join(f.split("_")[:2]) in impostors]
    sessions = [f for f in files if ("_").join(f.split("_")[:2]) in users]

    df = pd.DataFrame({
        "session": files,
        "split": None
    })
    df["user"] = df["session"].str.split("_", expand = True)[1].astype(int)
    
    training_set_len = int(0.6 * len(sessions))
    if training_set_len > len(users):
        idx_to_drop = df[df["session"].isin(impostor_sessions)].index
        guaranteed_sample = df.drop(idx_to_drop)
        guaranteed_sample = guaranteed_sample.groupby("user", group_keys = False).apply(
            lambda x: x.sample(n = 1, random_state = 19),
            include_groups = False
        )
        train_users = guaranteed_sample["session"].tolist()
        sessions = [s for s in sessions if s not in train_users]
        
        if training_set_len > len(train_users):
            rem_df = df.drop(guaranteed_sample.index)
            rem_df = rem_df.drop(idx_to_drop)
            
            add_num = training_set_len - len(train_users)
            add_samples = rem_df.sample(
                n = add_num,
                random_state = 19
            )
            train_users.extend(add_samples["session"].tolist())
            sessions = [s for s in sessions if s not in train_users]
        
        temp_users = sessions

    else:   
        train_users, temp_users = train_test_split(
        sessions, 
        train_size = 0.6,     
        random_state = 19
        )

    val_users, test_users = train_test_split(
    temp_users,
    test_size = 0.5,     
    random_state = 19
    )
    '''

    # assign splits
    df.loc[df["session"].isin(train_users), "split"] = "train"
    df.loc[df["session"].isin(val_users),   "split"] = "val"
    df.loc[df["session"].isin(test_users),  "split"] = "test"
    df.loc[df["session"].isin(impostor_sessions),  "split"] = "impostor"

    # save
    out_path = out_path / "split.csv"
    df.to_csv(out_path, index = False)

    return