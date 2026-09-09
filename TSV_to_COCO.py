import os, json
import pandas as pd

lsdat = pd.read_table('annot_tab.tsv')

df = lsdat[['image_name','image_width','image_height']].drop_duplicates().reset_index(drop=True)
df.columns = ['file','iwidth','iheight']
df['imgid'] = list(range(1, len(df['file'])+1))

lsdat = lsdat[['image_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'label_name']]
lsdat.columns = ['file','x','y','width','height','lab']

wd = pd.merge(lsdat, df, on='file', how='inner')
wd['id'] = list(range(1, len(wd['lab'])+1))

sc = list(wd['lab'].unique())
kats = pd.DataFrame(sc, columns=['lab']).sort_values(by=['lab'])
kats['category_id'] = list(range(1, len(sc)+1))

categories = []

for index, row in kats.iterrows():
    d = dict(id=row['category_id'], name=row['lab'], supercategory='cfu')
    categories.append(d)

wd = pd.merge(wd, kats, on='lab', how='inner')

images = []
for index, row in df.iterrows():
  d = dict(file_name=row['file'], width=row['iwidth'], height=row['iheight'], id=row['imgid'])
  images.append(d)

annotations = []
for index, row in wd.iterrows():
  d = dict(
  area=round(row['width'])*round(row['height']),
  iscrowd=False,
  bbox=[round(row['x']), round(row['y']), round(row['width']), round(row['height'])],
  category_id=row['category_id'],
  image_id=row['imgid'],
  id=row['id']
  )
  annotations.append(d)

out = dict(images=images, categories=categories, annotations=annotations, type=dict())

with open('annot_COCO.json', 'w') as outfile:
    json.dump(out, outfile)
