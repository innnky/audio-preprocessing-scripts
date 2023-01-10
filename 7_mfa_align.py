import requests
import os
mfa_zip = f'assets/mfa-opencpop-strict.zip'
dict_path = 'opencpop-strict.txt'
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
print(f'mfa align {segments_dir} {dict_path} {mfa_zip} {textgrids_dir} --beam 100 --clean --overwrite')
