import yaml

if __name__ == "__main__":
    chs = set()
    with open("MAE_pretraining/info_dataset/auditory.yaml") as f:
        audi = yaml.safe_load(f)

    for ch in audi["channel_list"]:
        chs.add(ch.lower())


    with open("MAE_pretraining/info_dataset/bci_comp_2b.yaml") as f:
        comp_b = yaml.safe_load(f)
    for ch in comp_b["channel_list"]:
        chs.add(ch.lower())

  
    
    with open("MAE_pretraining/info_dataset/LMI_C.yaml") as f:
        lmi = yaml.safe_load(f)
    for ch in lmi["channel_list"]:
        chs.add(ch.lower())
    
    with open("MAE_pretraining/info_dataset/p300.yaml") as f:
        p300 = yaml.safe_load(f)
    for ch in p300["channel_list"]:
        chs.add(ch.lower())
    with open("MAE_pretraining/info_dataset/seed2.yaml") as f:
        seed = yaml.safe_load(f)
    for ch in seed["channel_list"]:
        chs.add(ch.lower()) 
    
    print(len(chs))

    with open("/Users/sadeghemami/paper_1_code/MAE_pretraining/info_dataset/channel_info.yaml") as f:
        info = yaml.safe_load(f)

    with open("MAE_pretraining/info_dataset/bci_comp_2a.yaml") as f:
        comp_a = yaml.safe_load(f)

    chs_comp = [c.lower() for c in comp_a["channel_list"]]
    with open("downstream/info_dataset/upperlimb.yaml") as f:
        upper =yaml.safe_load(f)
    upper = [c.lower() for c in upper["channel_list"]]
    mapping = info["channels_mapping"]
    ch_map = [val.lower() for val,key in mapping.items()]
    print(len(mapping))
    for ch in upper:
       if ch not in ch_map:
           print(ch)

""""T9": 134,
    "TP9": 135,
    "T10": 136,
    "TP10": 137,
    "POO1": 138,
    "POO2": 139,
    "PPO1h": 140,
    "PPO2h": 141,"""

    