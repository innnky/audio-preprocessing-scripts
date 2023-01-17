import requests
import os
mfa_zip = f'assets/mfa-opencpop-strict.zip'
dict_path = 'assets/opencpop-strict.dict'
mfa_uri = 'https://diffsinger-1307911855.cos.ap-beijing.myqcloud.com/mfa/mfa-opencpop-strict.zip'
if not os.path.exists(mfa_zip):
    # Download
    print('Model not found, downloading...')
    with open(mfa_zip, 'wb') as f:
        f.write(requests.get(mfa_uri).content)
    print('Done.')
else:
    print('Pretrained model already exists.')

segments_dir = 'output'
textgrids_dir = 'textgrids'
os.makedirs(textgrids_dir, exist_ok=True)
print("使用预训练mfa声学模型则在安装mfa后手动执行下面命令")
print(f'mfa align {segments_dir} {dict_path} {mfa_zip} {textgrids_dir} --clean --overwrite')
print("不使用预训练模型（结果会更准确）则执行下面命令从零训练（请将数据拷贝至cpu比较好的电脑上训练mfa）")
print(f'mfa train {segments_dir} {dict_path} {mfa_zip} {textgrids_dir} --clean --overwrite')

