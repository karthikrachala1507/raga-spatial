import json

with open('outputs/json/test_detections.json') as f:
    data = json.load(f)

print('CLASS_248 APPEARANCES:')
print()
for w in data['windows']:
    for d in w['detections']:
        if d['class_idx'] == 248:
            print(str(w['start_sec']) + 's-' + str(w['end_sec']) + 's  confidence: ' + str(round(d['confidence']*100,1)) + '%')

print()
print('ALL UNKNOWN CLASSES FOUND:')
unknown = {}
for w in data['windows']:
    for d in w['detections']:
        if d['label'].startswith('Class_'):
            idx = d['class_idx']
            if idx not in unknown:
                unknown[idx] = []
            unknown[idx].append(round(d['confidence']*100,1))

for idx in sorted(unknown.keys()):
    confs = unknown[idx]
    print('Class_' + str(idx) + ' appeared ' + str(len(confs)) + ' windows  confidences: ' + str(confs))