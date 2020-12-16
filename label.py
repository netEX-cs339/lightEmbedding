import json
import pickle
import os.path as osp
from jsonvectorizer.utils import fopen

DATA_PATH = 'Data/json'
SAVE_PATH = 'Data/bin'

location_idx_list = ["location", "country"]
server_idx_list = ["p80", "http", "get", "headers", "server"]
certificate_idx_list = ["p110", "pop3", "starttls", "tls", "validation", "browser_trusted"]

location_dict = {"China": [], "Germany": [], "Japan": []}
server_dict = {"AkamaiGHost": [], "Apache": [], "Router": []}
certificate_dict = {True: [], False: []}
cnt = 0


def generate_dict(src, idx_list, target_dict, cnt):
    cur_level = src
    flag = True
    for item in idx_list:
        if item in cur_level.keys():
            cur_level = cur_level[item]
        else:
            flag = False

    if flag:
        for idx in target_dict.keys():
            if idx == cur_level:
                target_dict[idx].append(cnt)


with fopen(osp.join(DATA_PATH, 'sample10000.json')) as f:
    for line in f:
        doc = json.loads(line)

        generate_dict(doc, location_idx_list, location_dict, cnt)
        generate_dict(doc, certificate_idx_list, certificate_dict, cnt)

        cur_level = doc
        flag = True
        for item in server_idx_list:
            if item in cur_level.keys():
                cur_level = cur_level[item]
            else:
                flag = False

        if flag:
            if cur_level[:11] == "AkamaiGHost":
                server_dict["AkamaiGHost"].append(cnt)
            elif cur_level[:6] == "Apache":
                server_dict["Apache"].append(cnt)
            elif cur_level[:9] == "Microsoft":
                server_dict["Router"].append(cnt)
        ''''
        a = doc["p110"]["pop3"]["starttls"]["tls"]["validation"]["browser_trusted"]
        print(type(a))
        print(doc["p110"]["pop3"]["starttls"]["tls"]["validation"]["browser_trusted"])
        '''
        cnt = cnt + 1

print(location_dict)
print(server_dict)
print(certificate_dict)

for item in location_dict.keys():
    print(item, len(location_dict[item]))

for item in server_dict.keys():
    print(item, len(server_dict[item]))

for item in certificate_dict.keys():
    print(item, len(certificate_dict[item]))

with open(osp.join(SAVE_PATH, 'label_location.pkl'), 'wb') as f:
    pickle.dump(location_dict, f)

with open(osp.join(SAVE_PATH, 'label_server.pkl'), 'wb') as f:
    pickle.dump(server_dict, f)

with open(osp.join(SAVE_PATH, 'label_certificate.pkl'), 'wb') as f:
    pickle.dump(certificate_dict, f)
