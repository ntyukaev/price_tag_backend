from price_tag_backend.models import Barcode
import csv
from tqdm import tqdm

with open('barcode.csv', 'r', encoding='utf-8') as fin:
    dr = csv.DictReader(fin, delimiter='\t')
    to_db = [(i['ID'], i['UPCEAN'], i['Name']) for i in dr]

for inst in tqdm(to_db):
    try:
        b = Barcode(ID=inst[0], UPCEAN=inst[1], name=inst[2])
        print(inst)
        b.save()
    except ValueError:
        continue

