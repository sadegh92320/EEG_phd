import yaml

if __name__ == "__main__":
    chs = set()
    with open("MAE_pretraining/info_dataset/auditory.yaml") as f:
        audi = yaml.safe_load(f)

    for ch in audi["channel_list"]:
        chs.add(ch.lower())

    with open("MAE_pretraining/info_dataset/bci_comp_2a.yaml") as f:
        comp_a = yaml.safe_load(f)

    for ch in comp_a["channel_list"]:
        chs.add(ch.lower())

    with open("MAE_pretraining/info_dataset/bci_comp_2b.yaml") as f:
        comp_b = yaml.safe_load(f)
    for ch in comp_b["channel_list"]:
        chs.add(ch.lower())

    with open("MAE_pretraining/info_dataset/eeg_mi_bci.yaml") as f:
        mi_bci = yaml.safe_load(f)
    for ch in mi_bci["channel_list"]:
        chs.add(ch.lower())

    with open("MAE_pretraining/info_dataset/hgd.yaml") as f:
        hgd = yaml.safe_load(f)
    for ch in hgd["channel_list"]:
        chs.add(ch.lower())
    with open("MAE_pretraining/info_dataset/im_lab.yaml") as f:
        im_lab = yaml.safe_load(f)
    for ch in im_lab["channel_list"]:
        chs.add(ch.lower())
    with open("MAE_pretraining/info_dataset/LMI_C.yaml") as f:
        lmi = yaml.safe_load(f)
    for ch in lmi["channel_list"]:
        chs.add(ch.lower())
    with open("MAE_pretraining/info_dataset/online_bci_cla.yaml") as f:
        bci_cla = yaml.safe_load(f)
    for ch in bci_cla["channel_list"]:
        chs.add(ch.lower())
    with open("MAE_pretraining/info_dataset/p300.yaml") as f:
        p300 = yaml.safe_load(f)
    for ch in p300["channel_list"]:
        chs.add(ch.lower())
    with open("MAE_pretraining/info_dataset/seed2.yaml") as f:
        seed = yaml.safe_load(f)
    for ch in seed["channel_list"]:
        chs.add(ch.lower()) 
    with open("MAE_pretraining/info_dataset/ssvep.yaml") as f:
        ssvep = yaml.safe_load(f)

    for ch in ssvep["channel_list"]:
        chs.add(ch.lower()) 

    print(len(chs))

    with open("MAE_pretraining/info_dataset/channel_info.yaml") as f:
        info = yaml.safe_load(f)

    mapping = info["channels_mapping"]
    ch_map = [val.lower() for val,key in mapping.items()]
    print(len(mapping))
    for ch in chs:
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

    