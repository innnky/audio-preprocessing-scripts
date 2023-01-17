import requests
import os
mfa_zip = f'assets/mfa-opencpop-strict.zip'
dict_path = 'assets/opencpop-strict.dict'

segments_dir = 'output'
textgrids_dir = 'textgrids'
os.makedirs(textgrids_dir, exist_ok=True)
print("不使用预训练模型（结果会更准确）则执行下面命令从零训练（请将数据拷贝至cpu比较好的电脑上训练mfa）")
print(f'mfa train {segments_dir} {dict_path} {mfa_zip} {textgrids_dir} --clean --overwrite')

