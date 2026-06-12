import pandas as pd
from pathlib import Path

folder = Path(r'C:\Users\emre_\OneDrive\GitHub\CNN_fizzy\Code\recordings_v8')

new_headers = ['timestamp', 'roll' ,'pitch' ,'yaw' ,'acc_x','acc_y','acc_z','gyro_x', 'gyro_y', 'gyro_z', 'acc_mag', 'gyro_mag', 'motor_input', 'markers']

for f in folder.glob('*.csv'):
    df = pd.read_csv(f)
    if len(df.columns) != len(new_headers):
        print(f'skipped {f.name} (has {len(df.columns)} columns, expected {len(new_headers)})')
        continue
    df.columns = new_headers
    df.to_csv(f, index=False)
    print(f'fixed {f.name}')